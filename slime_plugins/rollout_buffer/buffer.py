import copy
import glob
import importlib.util
import json
import pathlib
import threading
import time
import traceback
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="Rollout Buffer Server", debug=True)


def default_is_valid_group(group_data, min_valid_group_size, task_type):
    instance_id, samples = group_data
    return len(samples) >= min_valid_group_size


def default_get_group_data_meta_info(temp_data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """
    Default implementation for getting meta information about the temporary data
    collected between get_batch calls.
    """
    if not temp_data:
        return {
            "total_samples": 0,
            "num_groups": 0,
            "avg_group_size": 0,
            "avg_reward": 0,
        }

    meta_info = {"total_samples": 0, "num_groups": len(temp_data)}

    all_rewards = []
    # Calculate per-group statistics
    for _instance_id, samples in temp_data.items():
        group_size = len(samples)
        group_rewards = [s["reward"] for s in samples]  # Calculate group reward standard deviation
        meta_info["total_samples"] += group_size
        all_rewards.extend(group_rewards)
    # Calculate global statistics
    meta_info["avg_group_size"] = meta_info["total_samples"] / meta_info["num_groups"]

    if all_rewards:
        meta_info["avg_reward"] = sum(all_rewards) / len(all_rewards)
    else:
        meta_info["avg_reward"] = 0
    return meta_info


def discover_generators():
    """
    Automatically discover generator modules in the generator directory.
    Returns a dictionary mapping task_type to module with run_rollout function.
    """
    generator_map = {}
    generator_dir = pathlib.Path(__file__).parent / "generator"

    # Find all files within generator_dir
    for file_path in glob.glob(str(generator_dir / "*.py")):
        if file_path.endswith("__init__.py"):
            continue

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("generator_module", file_path)
            if spec is None or spec.loader is None:
                print(f"Warning: Could not load spec for {file_path}")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if module has TASK_TYPE constant
            if not hasattr(module, "TASK_TYPE"):
                print(f"Warning: {file_path} does not define TASK_TYPE constant")
                continue

            # Check if module has run_rollout function
            if not hasattr(module, "run_rollout"):
                print(f"Warning: {file_path} does not define run_rollout function")
                continue

            task_type = module.TASK_TYPE
            generator_info = {
                "module": module,
                "file_path": file_path,
                "run_rollout": module.run_rollout,
            }

            # Check for optional functions and use defaults if not present
            for func_name in [
                "transform_group",
                "is_valid_group",
                "get_group_data_meta_info",
            ]:
                generator_info[func_name] = getattr(module, func_name, None)

            generator_map[task_type] = generator_info
            print(f"Discovered generator: {task_type} -> {file_path}")

        except Exception as e:
            print(f"Error loading generator from {file_path}: {str(e)}")
            continue

    return generator_map


@app.middleware("http")
async def set_body_size(request: Request, call_next):
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class BufferResponse(BaseModel):
    success: bool
    message: str = ""
    data: dict[str, Any] | None = None


class BufferQueue:
    def __init__(
        self,
        group_size,
        task_type="math",
        transform_group_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
    ):
        self.data = {}
        self.temp_data = {}
        self.group_timestamps = {}
        self.group_size = group_size
        self.task_type = task_type

        # Set up function handlers with defaults
        self.is_valid_group_func = is_valid_group_func or default_is_valid_group
        self.get_group_data_meta_info_func = get_group_data_meta_info_func or default_get_group_data_meta_info
        self.transform_group_func = transform_group_func or (lambda group, task_type: group)

    def append(self, item):
        instance_id = item["instance_id"]
        current_time = time.time()

        # Update timestamp for this group
        self.group_timestamps[instance_id] = current_time

        if instance_id not in self.temp_data:
            self.temp_data[instance_id] = [copy.deepcopy(item)]
        else:
            self.temp_data[instance_id].append(copy.deepcopy(item))

        if instance_id not in self.data:
            self.data[instance_id] = [item]
        else:
            self.data[instance_id].append(item)

    def _get_valid_groups_with_timeout(self, del_data=False):
        """Get valid groups including timeout-based groups"""
        valid_groups = {}
        timed_out_groups = {}
        finished_groups = []

        for instance_id, group_data in self.data.items():
            if self.is_valid_group_func((instance_id, group_data), self.group_size, self.task_type):
                valid_groups[instance_id] = group_data

        # Remove finished groups and timed out groups with insufficient data
        if del_data:
            for instance_id in finished_groups:
                self.data.pop(instance_id, None)
                self.group_timestamps.pop(instance_id, None)
                print(f"Removed finished group {instance_id}")

        # Combine normal valid groups and timeout groups
        all_valid_groups = {**valid_groups, **timed_out_groups}

        return all_valid_groups, finished_groups

    def get(self):
        output = {"data": [], "meta_info": {}}

        # Get meta information about temp data before processing
        meta_info = self.get_group_data_meta_info_func(self.temp_data)
        output["meta_info"] = meta_info

        valid_groups, finished_groups = self._get_valid_groups_with_timeout(del_data=True)
        output["meta_info"]["finished_groups"] = finished_groups

        print(f"meta info: {json.dumps(meta_info, indent=2)}")

        valid_groups = list(valid_groups.items())

        for instance_id, group in valid_groups:
            # First filter individual items
            transformed_group = self.transform_group_func((instance_id, group), self.task_type)
            output["data"].extend(transformed_group[1])

            if instance_id in self.data:
                self.data.pop(instance_id)

        return output

    def __len__(self):
        valid_groups, _ = self._get_valid_groups_with_timeout()
        num = sum([len(v) for v in valid_groups.values()])
        num_of_all_groups = sum([len(v) for v in self.data.values()])
        print(f"valid_groups: {len(valid_groups)}, num: {num}, num_of_all_groups: {num_of_all_groups}")
        return num


class RolloutBuffer:
    def __init__(
        self,
        group_size=16,
        task_type="math",
        transform_group_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
        rollout_job_id: str | None = None,
    ):
        self.buffer = BufferQueue(
            group_size=group_size,
            task_type=task_type,
            transform_group_func=transform_group_func,
            is_valid_group_func=is_valid_group_func,
            get_group_data_meta_info_func=get_group_data_meta_info_func,
        )
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.total_written = 0
        self.total_read = 0
        self.task_type = task_type
        self.rollout_job_id = rollout_job_id

    def write(self, data):
        with self.lock:
            item_job_id = data.get("rollout_job_id")
            if self.rollout_job_id and item_job_id is not None and item_job_id != self.rollout_job_id:
                print(f"Ignore stale rollout item from job {item_job_id}; " f"current job is {self.rollout_job_id}")
                return None
            self.buffer.append(data)
            self.total_written += 1
            self.not_empty.notify_all()
        return data

    def read(self):
        with self.not_empty:
            if len(self.buffer) == 0:
                return {"data": [], "meta_info": {}}

            # Don't clear temp_data for regular read operations
            result = self.buffer.get()
            self.total_read += len(result["data"])
            return result


buffer = RolloutBuffer()


class RolloutJob:
    def __init__(self, payload: dict[str, Any]):
        self.job_id = str(uuid.uuid4())
        self.payload = dict(payload)
        self.stop_event = threading.Event()
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.status = "starting"
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"rollout-buffer-{self.job_id[:8]}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout: float | None = None) -> bool:
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "alive": self.thread.is_alive(),
            "stop_requested": self.stop_event.is_set(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    def _run(self) -> None:
        self.status = "running"
        try:
            run_rollout(self.payload, stop_event=self.stop_event, rollout_job_id=self.job_id)
            self.status = "stopped" if self.stop_event.is_set() else "finished"
        except Exception as exc:
            self.status = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"Rollout job {self.job_id} failed: {self.error}")
            traceback.print_exc()
        finally:
            self.finished_at = time.time()


current_rollout_job: RolloutJob | None = None
rollout_job_lock = threading.RLock()


def _stop_current_rollout_job(wait: bool, timeout_sec: float | None) -> dict[str, Any]:
    with rollout_job_lock:
        job = current_rollout_job
    if job is None:
        return {"had_job": False, "stopped": True, "job": None}

    job.stop()
    stopped = True
    if wait and job.is_alive():
        stopped = job.join(timeout_sec)
    return {"had_job": True, "stopped": stopped, "job": job.snapshot()}


@app.post("/buffer/write", response_model=BufferResponse)
async def write_to_buffer(request: Request):
    try:
        data = await request.json()
        item = buffer.write(data)
        if item is None:
            return BufferResponse(
                success=False,
                message="Ignored stale rollout item",
                data={"data": [], "meta_info": "stale rollout job"},
            )
        return BufferResponse(
            success=True,
            message="Data has been successfully written to buffer",
            data={"data": [item], "meta_info": "write to buffer"},
        )
    except Exception as e:
        print(f"Write failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Write failed: {str(e)}") from e


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    items = buffer.read()

    if not items["data"]:
        return BufferResponse(
            success=False,
            message="No data available to read",
            data={"data": [], "meta_info": items["meta_info"]},
        )

    print(f"return {len(items['data'])} items and save them to local")
    buffer.buffer.temp_data = {}

    return BufferResponse(
        success=True,
        message=f"Successfully read {len(items['data'])} items",
        data=items,
    )


def run_rollout(
    data: dict,
    stop_event: threading.Event | None = None,
    rollout_job_id: str | None = None,
):
    global buffer
    # Auto-discover generators
    generator_map = discover_generators()

    task_type = data["task_type"]
    if task_type not in generator_map:
        print(f"Error: No generator found for task_type '{task_type}'")
        print(f"Available generators: {list(generator_map.keys())}")
        return

    generator_info = generator_map[task_type]
    print(f"Using generator: {generator_info['file_path']} for task_type: {task_type}")

    buffer = RolloutBuffer(
        group_size=int(data["num_repeat_per_sample"]),
        task_type=task_type,
        transform_group_func=generator_info.get("transform_group", None),
        is_valid_group_func=generator_info.get("is_valid_group"),
        get_group_data_meta_info_func=generator_info.get("get_group_data_meta_info"),
        rollout_job_id=rollout_job_id,
    )

    # Call the run_rollout function from the appropriate generator module
    generator_payload = dict(data)
    if stop_event is not None:
        generator_payload["_stop_event"] = stop_event
    if rollout_job_id is not None:
        generator_payload["_rollout_job_id"] = rollout_job_id
    generator_info["run_rollout"](generator_payload)
    print(f"Rollout completed successfully for task_type: {task_type}")


@app.post("/start_rollout")
async def start_rollout(request: Request):
    global current_rollout_job
    payload = await request.json()

    stop_timeout_sec = float(payload.get("stop_previous_timeout_sec", 60))
    stopped = _stop_current_rollout_job(wait=True, timeout_sec=stop_timeout_sec)
    if stopped["had_job"] and not stopped["stopped"]:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Previous rollout is still running after stop request; refusing to start a new one.",
                "previous": stopped["job"],
            },
        )

    job = RolloutJob(payload)
    with rollout_job_lock:
        current_rollout_job = job
    job.start()
    return {"message": "Rollout started", "rollout_job_id": job.job_id, "previous": stopped}


@app.post("/stop_rollout")
async def stop_rollout(request: Request):
    payload = await request.json()
    wait = bool(payload.get("wait", True))
    timeout_sec = payload.get("timeout_sec", 60)
    timeout_sec = None if timeout_sec is None else float(timeout_sec)
    return _stop_current_rollout_job(wait=wait, timeout_sec=timeout_sec)


@app.get("/rollout_status")
async def rollout_status():
    with rollout_job_lock:
        job = current_rollout_job
    if job is None:
        return {"job": None}
    return {"job": job.snapshot()}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8889,
        limit_concurrency=1000,  # Connection concurrency limit
        # limit_max_requests=1000000,  # Maximum request limit
        timeout_keep_alive=5,  # Keep-alive timeout,
    )
