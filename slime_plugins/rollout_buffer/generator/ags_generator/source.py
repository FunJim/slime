"""Prompt-data source used by the standalone rollout-buffer AGS generator."""

from __future__ import annotations

import copy
from argparse import Namespace

from slime.rollout.data_source import RolloutDataSource
from slime.utils.types import Sample


class AGSPromptSource:
    """Small wrapper around RolloutDataSource that yields one repeat at a time."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.data_source = RolloutDataSource(args)
        start_group = int(getattr(args, "rollout_start_group", 0) or 0)
        if start_group > 0 and self.data_source.dataset is not None:
            dataset_len = len(self.data_source.dataset)
            self.data_source.sample_offset = start_group % dataset_len if dataset_len else 0
            self.data_source.sample_group_index = start_group
            self.data_source.sample_index = start_group * int(args.n_samples_per_prompt)

    def get_groups(self, num_groups: int) -> list[list[Sample]]:
        groups = self.data_source.get_samples(num_groups)
        for group in groups:
            for sample in group:
                if sample.rollout_id is None:
                    sample.rollout_id = sample.index
        return groups

    def get_repeated_samples(self, num_groups: int, skip_instance_ids: list[str] | None = None) -> list[Sample]:
        skip = list(skip_instance_ids or [])
        samples: list[Sample] = []
        while len(samples) < num_groups * self.args.n_samples_per_prompt:
            groups = self.get_groups(num_groups)
            for group in groups:
                instance_id = _instance_id(group[0])
                for sample in group:
                    if instance_id in skip:
                        skip.remove(instance_id)
                        continue
                    samples.append(copy.deepcopy(sample))
                    if len(samples) >= num_groups * self.args.n_samples_per_prompt:
                        break
                if len(samples) >= num_groups * self.args.n_samples_per_prompt:
                    break
        return samples


def _instance_id(sample: Sample) -> str:
    metadata = sample.metadata or {}
    remote = metadata.get("remote_env_info") or {}
    label = sample.label if isinstance(sample.label, str) and len(sample.label) < 256 else None
    return str(metadata.get("instance_id") or remote.get("instance_id") or label or sample.index or "unknown")
