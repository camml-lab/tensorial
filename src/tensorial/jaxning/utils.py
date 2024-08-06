from collections.abc import Iterable, Mapping
import dataclasses
from typing import Generator, Optional, cast

import jax
import numpy as np
import torch


def _extract_batch_size(batch) -> Generator[Optional[int], None, None]:
    if isinstance(batch, (jax.Array, np.ndarray, torch.Tensor)):
        if batch.ndim == 1:
            yield 1
        else:
            yield batch.shape[0]

    elif isinstance(batch, (Iterable, Mapping)):
        if isinstance(batch, Mapping):
            batch = batch.values()

        for entry in batch:
            yield from _extract_batch_size(entry)

    elif dataclasses.is_dataclass(batch) and not isinstance(batch, type):
        batch = cast(dataclasses.dataclass, batch)
        for field in dataclasses.fields(batch):
            yield from _extract_batch_size(getattr(batch, field.name))

    else:
        yield None


def extract_batch_size(batch) -> int:
    error_msg = (
        "Could not determine batch size automatically.  You can provide this manually using "
        "`self.log(..., batch_size=size)`"
    )
    batch_size = None
    try:
        for size in _extract_batch_size(batch):
            if batch_size is None:
                batch_size = size
            elif size != batch_size:
                # TODO: Turn this into a warning
                print("Could not determine batch size unambiguously")
                break
    except RecursionError:
        raise RecursionError(error_msg)

    if batch_size is None:
        raise RuntimeError(error_msg)

    return batch_size


def sized_len(dataloader) -> Optional[int]:
    try:
        return len(dataloader)
    except (TypeError, NotImplementedError):
        return None
