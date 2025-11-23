from collections.abc import Iterator, Sequence
import functools
from typing import TYPE_CHECKING, Final

import beartype
import jaxtyping as jt
import jraph
import reax
from typing_extensions import override

from . import _batching, _common

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = ("GraphLoader",)


class GraphLoader(reax.DataLoader[_common.GraphsOrGraphsTuple, _common.GraphsOrGraphsTuple]):
    """Data loader for graphs"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        *datasets: jraph.GraphsTuple | Sequence[jraph.GraphsTuple] | None,
        batch_size: int = 1,
        shuffle: bool = False,
        pad: bool | None = None,
        padding: "gcnn.data.GraphPadding | None" = None,
        batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
        sampler: reax.data.Sampler = None,
    ):
        # Params
        self._batch_size: Final[int] = batch_size
        self._shuffle: Final[bool] = shuffle

        # State
        # If the graphs were supplied as GraphTuples then unbatch them to have a base sequence of
        # individual graphs per input
        self._dataset = tuple(
            jraph.unbatch_np(graphs) if isinstance(graphs, jraph.GraphsTuple) else graphs
            for graphs in datasets
        )
        self._sampler = self._create_sampler(self._dataset, batch_size, shuffle, sampler)

        if pad is None:
            pad = padding is not None

        create_batcher = functools.partial(
            _batching.GraphBatcher,
            batch_size=batch_size,
            shuffle=shuffle,
            pad=pad,
            padding=padding,
            mode=batch_mode,
        )
        self._batchers: "tuple[gcnn.data.GraphBatcher | None, ...]" = tuple(
            create_batcher(graph_batch) if graph_batch is not None else None
            for graph_batch in self._dataset
        )

    @staticmethod
    def _create_sampler(
        dataset, batch_size: int, shuffle: bool, sampler: reax.data.Sampler | None
    ) -> reax.data.Sampler:
        if sampler is not None:
            return sampler

        example = next(filter(lambda g: g is not None, dataset))
        return reax.data.samplers.create_sampler(example, batch_size=batch_size, shuffle=shuffle)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @override
    @property
    def dataset(self):
        return self._dataset

    @property
    def padding(self) -> "gcnn.data.GraphPadding":
        return self._batchers[0].padding

    @property
    def sampler(self) -> "reax.data.Sampler":
        """Access the index sampler used by the dataloader"""
        return self._sampler

    @override
    def __len__(self) -> int:
        return len(self.sampler)

    @override
    def __iter__(self) -> Iterator[tuple[jraph.GraphsTuple, ...]]:
        for idxs in self._sampler:
            batch_graphs = tuple(
                batcher.fetch(idxs) if batcher is not None else None for batcher in self._batchers
            )
            yield batch_graphs

    @override
    def with_new_sampler(self, sampler: "reax.data.Sampler") -> "GraphLoader":
        """Recreate the loader with the given index sampler"""
        return GraphLoader(
            *self._dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            padding=self.padding,
            sampler=sampler,
        )
