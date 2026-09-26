"""Information-preserving temporal stress for frozen reporting pipelines."""

from dataclasses import replace

import torch

from attcon.history_reporting import HistoryBatch


def insert_delay(data: HistoryBatch, extra_steps: int) -> HistoryBatch:
    if extra_steps < 0:
        raise ValueError("extra delay must be nonnegative")
    if extra_steps == 0:
        return data
    blank = data.events.new_zeros(len(data), extra_steps, data.events.shape[-1])
    return replace(data, events=torch.cat((data.events[:, :-1], blank, data.events[:, -1:]), dim=1))
