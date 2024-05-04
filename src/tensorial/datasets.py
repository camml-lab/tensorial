# -*- coding: utf-8 -*-
import itertools
from typing import Any, Callable, Iterable, Iterator, NamedTuple


class Batch(NamedTuple):
    inputs: Any
    targets: Any


def generate_batches(
    batch_size: int,
    inputs: Iterable,
    outputs: Iterable = None,
    batch_builder: Callable[[list], Any] = tuple,
    output_batch_builder: Callable[[list], Any] = None,
) -> Iterator[Batch]:
    """
    Generate batches from the given outputs.  This will yield batches of size `batch_size` as a
    tuple (inputs, targets).

    Functions to build each batch can optionally be supplied, these will be called as
    batch_builder(list(...)).
    """
    outputs = outputs if outputs is not None else itertools.cycle((None,))
    output_batch_builder = (
        output_batch_builder if output_batch_builder is not None else batch_builder
    )

    inp_iter = iter(inputs)
    out_iter = iter(outputs)
    while True:
        inps = list(itertools.islice(inp_iter, batch_size))
        outs = list(itertools.islice(out_iter, batch_size))
        if inps:
            yield Batch(inputs=batch_builder(inps), targets=output_batch_builder(outs))
        else:
            break
