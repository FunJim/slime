"""Regression tests for the PPO critic-resume behaviour of algo_args.sh.

Two defects made it into the first version of the PPO launcher, and both were
silent -- no error, just a run that re-trained consumed data or threw away the
critic's optimizer state. They also both live in argument resolution rather than
in training, so they are cheap to pin down here.

Defect 1, the run's start id
    slime takes the starting rollout from the CRITIC when one exists
    (slime/ray/placement_group.py, whose own TODO says the user must pin it),
    and Megatron forces iteration=0 whenever finetune is set. A cold-starting
    critic therefore reported rollout 1 even when the actor was resuming from
    rollout 50: the run restarted at 1 with rollout-50 actor weights, and
    rollout_manager.load(start_rollout_id - 1) rewound the dataset with it.
    algo_args.sh must emit --start-rollout-id in that combination.

Defect 2, the critic's optimizer state
    The resume branch of the generated YAML used to omit finetune /
    no_load_optim / no_load_rng and rely on inheriting them as False. But role
    overrides are applied on top of the args as slime_validate_args left them,
    and that function sets all three True whenever the ACTOR has no checkpoint.
    A critic with its own checkpoint then loaded its weights while discarding
    its Adam moments, its RNG state, and its LR schedule position. The resume
    branch must therefore write the three keys out explicitly.

The tests below drive the real algo_args.sh, so they fail if the script stops
emitting what the fixes depend on. They deliberately do not launch training.
"""

import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGO_ARGS = REPO_ROOT / "examples" / "coding_agent_rl_ags" / "algo_args.sh"


def _fake_megatron_checkpoint(path: Path, iteration: int) -> None:
    """Lay out the files slime and algo_args.sh treat as a resumable checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")
    iter_dir = path / f"iter_{iteration:07d}"
    iter_dir.mkdir(exist_ok=True)
    (iter_dir / ".metadata").write_bytes(b"")


def _run_algo_args(tmp_path: Path, *, actor_iteration=None, critic_iteration=None):
    """Source algo_args.sh with ADVANTAGE_ESTIMATOR=ppo and return what it built.

    Returns (algo_args list, parsed YAML dict, paths namespace).
    """
    exp = tmp_path / "exp"
    run_root = exp / "runs" / "r1"
    save = exp / "checkpoints"
    critic_save = exp / "critic_checkpoints"
    ref = tmp_path / "ref"
    run_root.mkdir(parents=True, exist_ok=True)
    save.mkdir(parents=True, exist_ok=True)
    _fake_megatron_checkpoint(ref, 0)
    if actor_iteration is not None:
        _fake_megatron_checkpoint(save, actor_iteration)
    if critic_iteration is not None:
        _fake_megatron_checkpoint(critic_save, critic_iteration)

    env = dict(
        os.environ,
        EXP=str(exp),
        RUN_ROOT=str(run_root),
        SAVE_DIR=str(save),
        CRITIC_SAVE_DIR=str(critic_save),
        REF_MODEL_PATH=str(ref),
        NUM_ROLLOUT="20",
        ADVANTAGE_ESTIMATOR="ppo",
    )
    result = subprocess.run(
        ["bash", "-c", f'set -e; source "{ALGO_ARGS}"; printf "%s\\n" "${{ALGO_ARGS[@]}}"'],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"algo_args.sh failed:\n{result.stdout}\n{result.stderr}"
    algo_args = [line for line in result.stdout.splitlines() if line and not line.startswith("PPO:")]

    with (run_root / "megatron_ppo.yaml").open() as handle:
        config = yaml.safe_load(handle)

    paths = Namespace(exp=exp, run_root=run_root, save=save, critic_save=critic_save, ref=ref)
    return algo_args, config, paths


def _critic_overrides(config: dict) -> dict:
    entries = [entry for entry in config["megatron"] if entry.get("role") == "critic"]
    assert len(entries) == 1, config
    return entries[0]["overrides"]


def _flag_value(algo_args: list, flag: str):
    if flag not in algo_args:
        return None
    return algo_args[algo_args.index(flag) + 1]


class TestPPOStartRolloutId:
    def test_pinned_when_critic_cold_starts_while_actor_resumes(self, tmp_path):
        """Defect 1: without this the run would restart from rollout 1."""
        algo_args, _, _ = _run_algo_args(tmp_path, actor_iteration=49, critic_iteration=None)

        assert _flag_value(algo_args, "--start-rollout-id") == "50"

    def test_not_pinned_when_critic_also_resumes(self, tmp_path):
        """Both roles carry a real iteration, so leave the derivation to slime."""
        algo_args, _, _ = _run_algo_args(tmp_path, actor_iteration=49, critic_iteration=49)

        assert "--start-rollout-id" not in algo_args

    def test_not_pinned_when_both_cold_start(self, tmp_path):
        """No actor resume point exists to pin to."""
        algo_args, _, _ = _run_algo_args(tmp_path, actor_iteration=None, critic_iteration=None)

        assert "--start-rollout-id" not in algo_args

    def test_explicit_override_wins(self, tmp_path):
        env_before = os.environ.get("START_ROLLOUT_ID")
        os.environ["START_ROLLOUT_ID"] = "7"
        try:
            algo_args, _, _ = _run_algo_args(tmp_path, actor_iteration=49, critic_iteration=None)
        finally:
            if env_before is None:
                os.environ.pop("START_ROLLOUT_ID", None)
            else:
                os.environ["START_ROLLOUT_ID"] = env_before

        assert _flag_value(algo_args, "--start-rollout-id") == "7"


class TestPPOCriticResumeFlags:
    def test_resume_writes_load_flags_explicitly(self, tmp_path):
        """Defect 2: inheriting these would discard the critic's optimizer state.

        They must be written out, because the base args they would otherwise be
        inherited from have all three set True whenever the actor cold-starts.
        """
        _, config, paths = _run_algo_args(tmp_path, actor_iteration=None, critic_iteration=24)
        overrides = _critic_overrides(config)

        assert overrides["finetune"] is False
        assert overrides["no_load_optim"] is False
        assert overrides["no_load_rng"] is False
        assert overrides["load"] == str(paths.critic_save)

    def test_cold_start_marks_finetune(self, tmp_path):
        """Base weights carry no optimizer or RNG state, and no value head."""
        _, config, paths = _run_algo_args(tmp_path, actor_iteration=None, critic_iteration=None)
        overrides = _critic_overrides(config)

        assert overrides["finetune"] is True
        assert overrides["no_load_optim"] is True
        assert overrides["no_load_rng"] is True
        assert overrides["load"] == str(paths.ref)


class TestPPOCriticRoleConfig:
    def test_critic_save_is_separate_from_actor(self, tmp_path):
        """Sharing --save would make both roles write the same iter_XXXXXXX."""
        _, config, paths = _run_algo_args(tmp_path, actor_iteration=None, critic_iteration=None)
        overrides = _critic_overrides(config)

        assert overrides["save"] == str(paths.critic_save)
        assert overrides["save"] != str(paths.save)

    def test_critic_lr_and_schedule(self, tmp_path):
        _, config, _ = _run_algo_args(tmp_path, actor_iteration=None, critic_iteration=None)
        overrides = _critic_overrides(config)

        # PyYAML reads a bare 1e-5 as a string; slime coerces it to the type of
        # the existing arg, so both spellings land as a float. Accept either.
        assert float(overrides["lr"]) == pytest.approx(1e-5)
        assert overrides["lr_decay_style"] == "constant"

    def test_yaml_shape_matches_parse_megatron_role_args(self, tmp_path):
        """The envelope parse_megatron_role_args asserts on."""
        _, config, _ = _run_algo_args(tmp_path, actor_iteration=None, critic_iteration=None)

        assert isinstance(config["megatron"], list)
        assert [entry["role"] for entry in config["megatron"]] == ["critic"]
        assert set(config["megatron"][0]) >= {"name", "role", "overrides"}


class TestPPOAlgoArgs:
    def test_gae_is_per_token(self, tmp_path):
        """gamma and lambd must stay 1.0.

        Reward lands on the final token and the response holds long spans of
        tool output the loss mask excludes; at lambd=1 GAE reduces to
        A_t = R - V(s_t), so no token's advantage is bootstrapped through
        positions the critic never trains on.
        """
        algo_args, _, _ = _run_algo_args(tmp_path, actor_iteration=None, critic_iteration=None)

        assert _flag_value(algo_args, "--gamma") == "1.0"
        assert _flag_value(algo_args, "--lambd") == "1.0"
        assert _flag_value(algo_args, "--advantage-estimator") == "ppo"
        assert _flag_value(algo_args, "--megatron-config-path") is not None

    def test_grpo_default_needs_no_role_config(self, tmp_path):
        """The default path must not have acquired PPO-only flags."""
        exp = tmp_path / "exp"
        (exp / "runs" / "r1").mkdir(parents=True)
        env = dict(
            os.environ,
            EXP=str(exp),
            RUN_ROOT=str(exp / "runs" / "r1"),
            SAVE_DIR=str(exp / "checkpoints"),
            REF_MODEL_PATH=str(tmp_path / "ref"),
            NUM_ROLLOUT="20",
        )
        env.pop("ADVANTAGE_ESTIMATOR", None)
        result = subprocess.run(
            ["bash", "-c", f'set -e; source "{ALGO_ARGS}"; printf "%s\\n" "${{ALGO_ARGS[@]}}"'],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        algo_args = [line for line in result.stdout.splitlines() if line]

        assert _flag_value(algo_args, "--advantage-estimator") == "grpo"
        for ppo_only in ("--megatron-config-path", "--num-critic-only-steps", "--gamma", "--lambd"):
            assert ppo_only not in algo_args
        assert not (exp / "runs" / "r1" / "megatron_ppo.yaml").exists()

    def test_unsupported_estimator_aborts_the_caller(self, tmp_path):
        """algo_args.sh is sourced, so its exit must stop the launcher."""
        exp = tmp_path / "exp"
        (exp / "runs" / "r1").mkdir(parents=True)
        env = dict(
            os.environ,
            EXP=str(exp),
            RUN_ROOT=str(exp / "runs" / "r1"),
            SAVE_DIR=str(exp / "checkpoints"),
            REF_MODEL_PATH=str(tmp_path / "ref"),
            NUM_ROLLOUT="20",
            ADVANTAGE_ESTIMATOR="gspo",
        )
        result = subprocess.run(
            ["bash", "-c", f'set -e; source "{ALGO_ARGS}"; echo REACHED_CALLER'],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "REACHED_CALLER" not in result.stdout
        assert "unsupported ADVANTAGE_ESTIMATOR" in result.stderr
