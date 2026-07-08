import asyncio
import time
from typing import Any

import aiohttp
import requests
import wandb
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

__all__ = ["generate_rollout"]


# Global variables for evaluation
TOKENIZER = None
START_ROLLOUT = True


def select_rollout_data(args, results, need_length):
    """
    Select the most recent groups when there are too many samples.
    Groups all samples by instance_id, sorts groups by timestamp.

    Args:
        args: Arguments containing configuration
        results: List of rollout data items with timestamps

    Returns:
        Selected samples from the newest groups based on timestamp cutoff
    """
    if not results:
        return results

    # Group samples by instance_id
    groups = {}
    for item in results:
        assert "instance_id" in item, "instance_id must be in item"
        instance_id = item["instance_id"]
        if instance_id not in groups:
            groups[instance_id] = []
        groups[instance_id].append(item)

    print(f"📊 Total groups: {len(groups)}, total samples: {len(results)}")

    # Return grouped records even when there is no over-collection; downstream
    # expects prompt-group shape, not a flat record list.
    if len(groups) <= need_length:
        return list(groups.values())

    # Get timestamp for each group (use the latest timestamp in the group)
    def get_group_timestamp(group_items):
        timestamps = []
        for item in group_items:
            if "timestamp" in item:
                timestamps.append(float(item["timestamp"]))
            elif "extra_info" in item and "timestamp" in item["extra_info"]:
                timestamps.append(float(item["extra_info"]["timestamp"]))
        return max(timestamps) if timestamps else 0

    # Create list of (group_id, timestamp, samples) and sort by timestamp
    group_data = []
    for group_id, group_items in groups.items():
        group_timestamp = get_group_timestamp(group_items)
        group_data.append((group_id, group_timestamp, group_items))

    # Sort groups by timestamp (newest first)
    group_data.sort(key=lambda x: x[1], reverse=True)

    selected_groups = group_data[:need_length]

    # Flatten selected groups back to sample list
    selected_results = []
    for _group_id, _timestamp, group_items in selected_groups:
        selected_results.append(group_items)

    # Statistics for monitoring
    if selected_groups:
        newest_ts = selected_groups[0][1]
        oldest_ts = selected_groups[-1][1]
        print(
            f"📈 Selected {len(selected_groups)} groups with {len(selected_results)*args.n_samples_per_prompt} samples"
        )
        print(f"📈 Group timestamp range: {oldest_ts:.2f} to {newest_ts:.2f}")
        print(f"📈 Time span: {newest_ts - oldest_ts:.2f} seconds")

    return selected_results


def log_raw_info(args, all_meta_info, rollout_id):
    if not all_meta_info:
        return

    final_meta_info = _merge_rollout_meta_info(all_meta_info)
    if final_meta_info.get("total_samples", 0) <= 0:
        print(f"no filter rollout log {rollout_id}: {final_meta_info}")
        return

    try:
        step = (
            rollout_id
            if not args.wandb_always_use_train_step
            else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        log_dict = _flatten_numeric_metrics("rollout/no_filter", final_meta_info)
        if getattr(args, "rollout_task_type", None) == "ags":
            log_dict.update(_flatten_numeric_metrics("rollout/ags", final_meta_info))
        log_dict["rollout/step"] = step
        if args.use_wandb:
            wandb.log(log_dict)

        if args.use_tensorboard:
            from slime.utils.tensorboard_utils import _TensorboardAdapter

            tb = _TensorboardAdapter(args)
            tb.log(data=log_dict, step=step)
        print(f"no filter rollout log {rollout_id}: {log_dict}")
    except Exception as e:
        print(f"Failed to log rollout metrics: {e}")
        print(f"no filter rollout log {rollout_id}: {final_meta_info}")


def _merge_rollout_meta_info(all_meta_info: list[dict[str, Any]]) -> dict[str, Any]:
    total_samples = sum(int(meta.get("total_samples", 0) or 0) for meta in all_meta_info)
    merged: dict[str, Any] = {"total_samples": total_samples}
    if total_samples <= 0:
        return merged

    merged["num_groups"] = sum(int(meta.get("num_groups", 0) or 0) for meta in all_meta_info)

    if any("total_rollouts" in meta for meta in all_meta_info):
        merged["total_rollouts"] = sum(
            int(meta.get("total_rollouts", meta.get("total_samples", 0)) or 0) for meta in all_meta_info
        )

    for key in ["nonzero_reward_samples", "solved_samples"]:
        if any(key in meta for meta in all_meta_info):
            merged[key] = sum(int(meta.get(key, 0) or 0) for meta in all_meta_info)

    weighted_avg_keys = [
        "avg_reward",
        "solve_rate",
        "nonzero_reward_rate",
        "completed_rate",
        "abort_rate",
        "artifact_complete_rate",
    ]
    for key in weighted_avg_keys:
        if not any(key in meta for meta in all_meta_info):
            continue
        weighted_sum = sum(
            float(meta[key]) * int(meta.get("total_samples", 0) or 0)
            for meta in all_meta_info
            if key in meta and meta.get("total_samples", 0)
        )
        merged[key] = weighted_sum / total_samples

    num_groups = int(merged.get("num_groups", 0) or 0)
    if "total_rollouts" in merged:
        merged["avg_group_size"] = int(merged["total_rollouts"]) / num_groups if num_groups else 0
        merged["avg_samples_per_group"] = total_samples / num_groups if num_groups else 0
    elif any("avg_group_size" in meta for meta in all_meta_info):
        weighted_sum = sum(
            float(meta["avg_group_size"]) * int(meta.get("total_samples", 0) or 0)
            for meta in all_meta_info
            if "avg_group_size" in meta and meta.get("total_samples", 0)
        )
        merged["avg_group_size"] = weighted_sum / total_samples

    for key in ["status_counts", "artifact_counts"]:
        if any(key in meta for meta in all_meta_info):
            merged[key] = _sum_nested_counts(meta.get(key, {}) for meta in all_meta_info)

    performance_items = [meta.get("performance", {}) for meta in all_meta_info]
    if any(performance_items):
        elapsed_sec_values = [
            float(value)
            for item in performance_items
            if isinstance(item, dict)
            for value in item.get("elapsed_sec_values", [])
        ]
        merged["performance"] = {
            "rollout_concurrency": _max_nested(performance_items, "rollout_concurrency"),
            "avg_elapsed_sec": (
                sum(elapsed_sec_values) / len(elapsed_sec_values)
                if elapsed_sec_values
                else _weighted_average_nested(performance_items, all_meta_info, "avg_elapsed_sec")
            ),
            "p50_elapsed_sec": (
                _percentile(elapsed_sec_values, 0.50)
                if elapsed_sec_values
                else _max_nested(performance_items, "p50_elapsed_sec")
            ),
            "p95_elapsed_sec": (
                _percentile(elapsed_sec_values, 0.95)
                if elapsed_sec_values
                else _max_nested(performance_items, "p95_elapsed_sec")
            ),
            "max_elapsed_sec": (
                max(elapsed_sec_values) if elapsed_sec_values else _max_nested(performance_items, "max_elapsed_sec")
            ),
            "agent_exit_nonzero_count": sum(
                int(item.get("agent_exit_nonzero_count", 0) or 0) for item in performance_items
            ),
            "applied_cleanly_count": sum(int(item.get("applied_cleanly_count", 0) or 0) for item in performance_items),
        }
    return merged


def _sum_nested_counts(dicts) -> dict[str, int]:
    output: dict[str, int] = {}
    for data in dicts:
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, int | float):
                output[str(key)] = output.get(str(key), 0) + int(value)
    return output


def _weighted_average_nested(items: list[dict[str, Any]], all_meta_info: list[dict[str, Any]], key: str) -> float:
    weighted_sum = 0.0
    total_weight = 0
    for item, meta in zip(items, all_meta_info, strict=False):
        if key not in item:
            continue
        weight = int(meta.get("total_samples", 0) or 0)
        weighted_sum += float(item[key]) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0


def _max_nested(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items if isinstance(item, dict) and key in item]
    return max(values) if values else 0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _flatten_numeric_metrics(prefix: str, data: dict[str, Any]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for key, value in data.items():
        metric_key = f"{prefix}/{key}"
        if isinstance(value, bool):
            metrics[metric_key] = int(value)
        elif isinstance(value, int | float):
            metrics[metric_key] = value
        elif isinstance(value, dict):
            metrics.update(_flatten_numeric_metrics(metric_key, value))
    return metrics


async def get_rollout_data(api_base_url: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.post(
                f"{api_base_url}/get_rollout_data", json={}, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response.raise_for_status()
                resp_json = await response.json()
                if resp_json["success"]:
                    break
            await asyncio.sleep(3)
            if time.time() - start_time > 30:
                print("rollout data is not ready, have been waiting for 30 seconds")
                # Reset start_time to continue waiting or handle timeout differently
                start_time = time.time()  # Or raise an exception, or return empty list

        data = resp_json["data"]
        meta_info = {}
        if isinstance(data, list):
            if "data" in data[0]:
                data = [item["data"] for item in data]
        elif isinstance(data, dict):
            if "data" in data:
                meta_info = data["meta_info"]
                data = data["data"]
        print(f"Meta info: {meta_info}")
        required_keys = {"uid", "instance_id", "messages", "reward", "extra_info"}
        for item in data:
            if not required_keys.issubset(item.keys()):
                raise ValueError(f"Missing required keys in response item: {item}")

        return data, meta_info


def start_rollout(api_base_url: str, args, metadata):
    url = f"{api_base_url}/start_rollout"
    print(f"metadata: {metadata}")
    finished_groups_instance_id_list = [item for sublist in metadata.values() for item in sublist]
    payload = {
        "num_process": str(getattr(args, "rollout_num_process", 100)),
        "num_epoch": str(args.num_epoch or 3),
        "remote_engine_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        "remote_buffer_url": args.rollout_buffer_url,
        "task_type": args.rollout_task_type,
        "input_file": args.prompt_data,
        "num_repeat_per_sample": str(args.n_samples_per_prompt),
        "max_tokens": str(args.rollout_max_response_len),
        "sampling_params": {
            "max_tokens": args.rollout_max_response_len,
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
        },
        "tokenizer_path": args.hf_checkpoint,
        "input_key": getattr(args, "input_key", "prompt"),
        "label_key": getattr(args, "label_key", "label"),
        "metadata_key": getattr(args, "metadata_key", "metadata"),
        "tool_key": getattr(args, "tool_key", None),
        "apply_chat_template": getattr(args, "apply_chat_template", False),
        "apply_chat_template_kwargs": getattr(args, "apply_chat_template_kwargs", {}) or {},
        "rollout_batch_size": args.rollout_batch_size,
        "rollout_max_context_len": getattr(args, "rollout_max_context_len", 0),
        "rollout_seed": getattr(args, "rollout_seed", 42),
        "rollout_shuffle": getattr(args, "rollout_shuffle", False),
        "sglang_tool_call_parser": getattr(args, "sglang_tool_call_parser", None),
        "sglang_reasoning_parser": getattr(args, "sglang_reasoning_parser", None),
        "skip_instance_ids": finished_groups_instance_id_list,
    }
    print("start rollout with payload: ", payload)

    while True:
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(f"[start_rollout] Success: {data}")
            return data
        except Exception as e:
            print(f"[start_rollout] Failed to send rollout config: {e}")


async def generate_rollout_async(args, rollout_id: int, data_buffer, evaluation: bool = False) -> dict[str, Any]:

    global START_ROLLOUT
    if evaluation:
        raise NotImplementedError("Evaluation rollout is not implemented")

    if START_ROLLOUT:
        metadata = data_buffer.get_metadata()
        start_inform = start_rollout(args.rollout_buffer_url, args, metadata)
        print(f"start rollout with payload: {start_inform}")
        print(f"start rollout id: {rollout_id}")
        START_ROLLOUT = False

    data_number_to_fetch = args.rollout_batch_size * args.n_samples_per_prompt - data_buffer.get_buffer_length()
    if data_number_to_fetch <= 0:
        print(
            f"❕buffer length: {data_buffer.get_buffer_length()}, buffer has enough data, return {args.rollout_batch_size} prompts"
        )
        return data_buffer.get_samples(args.rollout_batch_size)
    assert (
        data_number_to_fetch % args.n_samples_per_prompt == 0
    ), "data_number_to_fetch must be a multiple of n_samples_per_prompt"
    print(f"INFO: buffer length: {data_buffer.get_buffer_length()}, data_number_to_fetch: {data_number_to_fetch}")
    base_url = args.rollout_buffer_url
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    retry_times = 0
    results = []
    all_meta_info = []

    if args.fetch_trajectory_retry_times == -1:
        print(
            "⚠️  [get_rollout_data] Fetch trajectory retry times set to -1, will retry indefinitely until sufficient data is collected"
        )
    while args.fetch_trajectory_retry_times == -1 or retry_times < args.fetch_trajectory_retry_times:
        try:
            while len(results) < data_number_to_fetch:
                time.sleep(5)
                data, meta_info = await get_rollout_data(api_base_url=base_url)
                results.extend(data)
                if meta_info:
                    all_meta_info.append(meta_info)
                print(f"get rollout data with length: {len(results)}")
            break
        except Exception as err:
            print(f"[get_rollout_data] Failed to get rollout data: {err}, retry times: {retry_times}")
            retry_times += 1

    log_raw_info(args, all_meta_info, rollout_id)

    # Apply group-based data selection if there are too many samples
    results = select_rollout_data(args, results, data_number_to_fetch // args.n_samples_per_prompt)

    if len(all_meta_info) > 0 and "finished_groups" in all_meta_info[0]:
        finished_groups_instance_id_list = []
        for item in all_meta_info:
            finished_groups_instance_id_list.extend(item["finished_groups"])

        data_buffer.update_metadata({str(rollout_id): finished_groups_instance_id_list})

    print("finally get rollout data with length: ", len(results))
    sample_results = []

    for _i, group_record in enumerate(results):
        group_results = []
        for record in group_record:
            if "samples" in record:
                compact_samples = [Sample.from_dict(item) for item in record["samples"]]
                group_results.append(compact_samples)
                continue

            oai_messages = record["messages"]

            mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
            token_ids, loss_mask = mask_generator.get_loss_mask(oai_messages)
            response_length = mask_generator.get_response_lengths([loss_mask])[0]

            loss_mask = loss_mask[-response_length:]

            group_results.append(
                Sample(
                    index=record["instance_id"],
                    prompt=record["uid"],
                    tokens=token_ids,
                    response_length=response_length,
                    reward=record["reward"],
                    status=(
                        Sample.Status.COMPLETED
                        if "finish_reason" not in record["extra_info"]
                        or record["extra_info"]["finish_reason"] != "length"
                        else Sample.Status.TRUNCATED
                    ),
                    loss_mask=loss_mask,
                    metadata={**record["extra_info"]},
                )
            )
        sample_results.append(group_results)

    data_buffer.add_samples(sample_results)
    final_return_results = data_buffer.get_samples(args.rollout_batch_size)  # type: ignore

    return final_return_results


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Generate rollout for both training and evaluation."""
    return run(generate_rollout_async(args, rollout_id, data_buffer, evaluation))
