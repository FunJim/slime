"""Per-sample AGS coding-agent rollout implementation."""

from __future__ import annotations

import asyncio
import copy
import logging
import secrets
import time
import traceback
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from slime.utils.types import Sample

from .adapter_service import AdapterService, RemoteAdapterService
from .ags_sandbox import AGSSandbox
from .artifacts import ArtifactWriter, sample_artifact_id
from .config import AGSGeneratorConfig
from .harnesses import resolve_agent
from .sampling import normalize_sampling_params
from .swe_task import evaluate, get_metadata, git_diff, prepare_workspace
from .weave_trace import AGSWeaveTrace

logger = logging.getLogger(__name__)


class AGSRolloutRunner:
    def __init__(
        self,
        args: Namespace,
        config: AGSGeneratorConfig | None = None,
        *,
        use_remote_adapter: bool = False,
    ) -> None:
        self.args = args
        self.config = config or AGSGeneratorConfig.from_env()
        self.harness_cls, self.adapter_cls = resolve_agent(self.config.agent_name)
        if use_remote_adapter:
            self.adapter_service = RemoteAdapterService(args, self.config)
        else:
            self.adapter_service = AdapterService(args, self.config, self.adapter_cls)
        self.artifacts = ArtifactWriter(self.config.artifact_dir)
        self.weave_trace = AGSWeaveTrace(
            args,
            self.adapter_service.tokenizer,
            agent=self.config.agent_name,
            enable_token2text=self.config.enable_token2text,
        )
        self._boot_sem = asyncio.Semaphore(self.config.boot_concurrency)

    async def generate(self, base_sample: Sample, sampling_params: dict) -> list[Sample]:
        md = get_metadata(base_sample)
        instance_id = md["instance_id"]
        base_sample = copy.deepcopy(base_sample)
        session_id = _session_id(base_sample, instance_id)
        base_sample.session_id = session_id
        artifact_id = sample_artifact_id(instance_id, base_sample)
        normalized_sampling = normalize_sampling_params(sampling_params)
        trace_call = self.weave_trace.start_rollout(
            instance_id=instance_id,
            session_id=session_id,
            sample=base_sample,
            sampling_params=normalized_sampling,
            agent=self.config.agent_name,
        )
        if not md["image"] or not md["workdir"]:
            samples = self._abort_result(base_sample, "missing_image_or_workdir", instance_id)
            self.weave_trace.finish_rollout(trace_call, samples=samples)
            return samples
        t0 = time.time()
        session_opened = False
        trajectory_path = None
        evaluation_args = {
            "image": md["image"],
            "workdir": md["workdir"],
            "swepro": md["swepro"],
            "eval_cmd": md["eval_cmd"],
            "f2p_script": md["f2p_script"],
            "pre_commands": md["pre_commands"],
            "eval_bootstrap_cmd": self.config.eval_bootstrap_cmd,
            "timeout_sec": self.config.eval_timeout_sec,
        }
        try:
            self.adapter_service.adapter.open_session(
                session_id,
                sampling_defaults=normalized_sampling,
                max_context_tokens=self.adapter_service.max_context_len,
            )
            session_opened = True
            async with asyncio.timeout(self.config.rollout_guard_sec):
                async with self._boot_agent_sandbox(md["image"], instance_id) as sb:
                    await prepare_workspace(sb, md["workdir"], md)
                    agent_exit_code = await self.harness_cls().run(
                        sb,
                        workdir=md["workdir"],
                        session_id=session_id,
                        adapter_url=self.adapter_service.adapter_url,
                        time_budget_sec=self.config.agent_time_budget_sec,
                        prompt=self.config.prompt,
                    )
                    trajectory_path = await self.artifacts.dump_trajectory(sb, md["workdir"], artifact_id)
                    diff_text = await git_diff(sb, md["workdir"])
                    patch_path = self.artifacts.dump_patch(diff_text, artifact_id)
                    if not self.config.eval_isolated_sandbox:
                        reward, applied_cleanly = await evaluate(
                            sandbox=sb,
                            diff_text=diff_text,
                            **evaluation_args,
                        )

                if self.config.eval_isolated_sandbox:
                    reward, applied_cleanly = await evaluate(
                        sandbox=None,
                        diff_text=diff_text,
                        **evaluation_args,
                    )
                samples = await self.adapter_service.adapter.finish_session(
                    session_id,
                    base_sample=base_sample,
                    reward=float(reward),
                    extra_metadata={
                        "grading_solved": float(reward) == 1.0,
                        "instance_id": instance_id,
                    },
                )
                if not samples:
                    samples = self._abort_result(base_sample, "adapter_session_empty", instance_id)
                    self.weave_trace.finish_rollout(trace_call, samples=samples, trajectory_path=trajectory_path)
                    return samples

                rollout_path = self.artifacts.dump_rollout(
                    {
                        "instance_id": instance_id,
                        "session_id": session_id,
                        "agent": self.config.agent_name,
                        "reward": float(reward),
                        "applied_cleanly": bool(applied_cleanly),
                        "eval_isolated_sandbox": self.config.eval_isolated_sandbox,
                        "agent_exit_code": agent_exit_code,
                        "elapsed_sec": time.time() - t0,
                        "num_samples": len(samples),
                        "patch_path": patch_path,
                        "trajectory_path": trajectory_path,
                    },
                    artifact_id,
                )
                elapsed_sec = time.time() - t0
                for sample in samples:
                    sample.metadata = {
                        **(sample.metadata or {}),
                        "agent": self.config.agent_name,
                        "agent_exit_code": agent_exit_code,
                        "applied_cleanly": bool(applied_cleanly),
                        "eval_isolated_sandbox": self.config.eval_isolated_sandbox,
                        "trajectory_path": trajectory_path,
                        "patch_path": patch_path,
                        "rollout_dump_path": rollout_path,
                        "ags_elapsed_sec": elapsed_sec,
                        "ags_num_samples": len(samples),
                        "ags_rollout_concurrency": self.config.rollout_concurrency,
                    }
                logger.info(
                    "[ags_generator] %s: reward=%.2f applied=%s eval_isolated=%s exit=%s elapsed=%.1fs segments=%d",
                    instance_id,
                    float(reward),
                    bool(applied_cleanly),
                    self.config.eval_isolated_sandbox,
                    agent_exit_code,
                    elapsed_sec,
                    len(samples),
                )
                self.weave_trace.finish_rollout(
                    trace_call,
                    samples=samples,
                    trajectory_path=trajectory_path,
                    output={
                        "reward": float(reward),
                        "applied_cleanly": bool(applied_cleanly),
                        "agent_exit_code": agent_exit_code,
                        "elapsed_sec": elapsed_sec,
                        "patch_path": patch_path,
                        "rollout_dump_path": rollout_path,
                    },
                )
                return samples
        except asyncio.TimeoutError:
            _log_timeout_diagnostic(t0, instance_id, self.config.rollout_guard_sec)
            samples = self._abort_result(base_sample, "wall_clock_timeout", instance_id)
            self.weave_trace.finish_rollout(trace_call, samples=samples, trajectory_path=trajectory_path)
            return samples
        except Exception as exc:
            logger.warning("[ags_generator] %s: rollout failed: %s\n%s", instance_id, exc, traceback.format_exc())
            samples = self._abort_result(base_sample, f"exception:{type(exc).__name__}", instance_id)
            self.weave_trace.finish_rollout(
                trace_call,
                samples=samples,
                trajectory_path=trajectory_path,
                exception=exc,
            )
            return samples
        finally:
            if session_opened:
                try:
                    await self.adapter_service.adapter.drop_session(session_id)
                except Exception:
                    logger.warning(
                        "[ags_generator] %s: failed to drop session %s\n%s",
                        instance_id,
                        session_id,
                        traceback.format_exc(),
                    )

    @asynccontextmanager
    async def _boot_agent_sandbox(self, image: str, instance_id: str) -> AsyncIterator[AGSSandbox]:
        sb = None
        last_err: Exception | None = None
        for attempt in range(self.config.boot_retries):
            cand = AGSSandbox(image)
            try:
                async with self._boot_sem:
                    await cand.__aenter__()
                    try:
                        await self.harness_cls().install_cli(cand)
                    except BaseException:
                        await cand.__aexit__(None, None, None)
                        raise
                sb = cand
                break
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "[ags_generator] %s: AGS provision attempt %d/%d failed: %s: %s",
                    instance_id,
                    attempt + 1,
                    self.config.boot_retries,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                await asyncio.sleep(1 + attempt)
        if sb is None:
            assert last_err is not None
            raise last_err
        try:
            yield sb
        finally:
            await sb.__aexit__(None, None, None)

    def _abort_result(self, sample: Sample, reason: str, instance_id: str) -> list[Sample]:
        sample = copy.deepcopy(sample)
        sample.tokens = [0, 0]
        sample.response = ""
        sample.response_length = 1
        sample.loss_mask = [0]
        sample.rollout_log_probs = [0.0]
        sample.reward = 0.0
        sample.remove_sample = True
        sample.status = Sample.Status.ABORTED
        sample.metadata = {**(sample.metadata or {}), "abort_reason": reason, "instance_id": instance_id}
        logger.warning("[ags_generator] %s aborted: %s", instance_id, reason)
        return [sample]


def _session_id(sample: Sample, instance_id: str) -> str:
    """Return a fresh adapter session id for one AGS attempt.

    RolloutDataSource can hand out deep copies of Samples that already carry a
    session_id, and failed/partial reruns can also revisit the same
    (instance_id, index, group_index).  Adapter sessions are process-global for
    one generator run, so every AGS attempt must get a unique id instead of
    reusing the sample's existing session_id.
    """

    parts = ["cagent", instance_id]
    if sample.index is not None:
        parts.append(str(sample.index))
    if sample.group_index is not None:
        parts.append(str(sample.group_index))
    parts.append(secrets.token_hex(4))
    return "-".join(parts)


def _log_timeout_diagnostic(t0: float, instance_id: str, guard_sec: int) -> None:
    try:
        elapsed = time.time() - t0
        pending = [task for task in asyncio.all_tasks() if not task.done()]
        stuck = []
        for task in pending[:5]:
            coro = getattr(task, "_coro", None)
            stuck.append(getattr(coro, "__qualname__", repr(coro)))
        logger.warning(
            "[ags_generator] %s: wall_clock_timeout after %.1fs (guard=%ds); %d tasks pending; sample=%s",
            instance_id,
            elapsed,
            guard_sec,
            len(pending),
            stuck,
        )
    except Exception:
        pass
