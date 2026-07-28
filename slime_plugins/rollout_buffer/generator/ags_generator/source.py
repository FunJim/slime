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
        dataset = self.data_source.dataset
        if start_group > 0 and dataset is not None and len(dataset) > 0:
            # There is no generator-side checkpoint: the trainer identifies the
            # position by absolute group index (rollout_id * rollout_batch_size),
            # so seek to it. Group index g lives at offset g % len in epoch
            # g // len, mirroring how RolloutDataSource.get_samples advances the
            # offset by one per group and bumps the epoch on each wraparound.
            dataset_len = len(dataset)
            self.data_source.sample_offset = start_group % dataset_len
            self.data_source.epoch_id = start_group // dataset_len
            self.data_source.sample_group_index = start_group
            self.data_source.sample_index = start_group * int(args.n_samples_per_prompt)
            if args.rollout_shuffle:
                # Each epoch has its own seeded permutation; without this the
                # samples stay on epoch 0's order after a wraparound and the
                # prompt sequence diverges from the uninterrupted run.
                dataset.shuffle(self.data_source.epoch_id)

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
