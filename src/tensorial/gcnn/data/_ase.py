"""Module for loading ase.Atoms objects as graphs"""

from collections.abc import Sequence

from typing import TYPE_CHECKING, Any, Final

import jraph

from ... import data, utils
from .. import atomic
from . import _common, _datamodule

if TYPE_CHECKING:
    import jax

    from tensorial import gcnn

__all__ = (
    "AseDataLoader",
    "ase_graph_module_from_datasets",
    "ase_graph_module_kfold",
    "ase_graph_module_random_split",
)


PathSpec = str | Sequence[str]
SplitSpec = PathSpec | dict[str, PathSpec]


def _init_kwargs(limit: int | None, read_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    if read_kwargs is None:
        read_kwargs = {"index": f":{limit}" if limit is not None else ":"}
    else:
        if limit is not None:
            read_kwargs["index"] = f":{limit}"
        elif "index" not in read_kwargs:
            read_kwargs["index"] = ":"
    return read_kwargs


class AseGraphs(Sequence[jraph.GraphsTuple]):
    """A sequence of ASE structures served as lazily-built graphs.

    Wraps an already-loaded list of :class:`ase.Atoms` objects (e.g. produced by
    :class:`AseDataFetcher`).  Each structure is converted to a
    :class:`jraph.GraphsTuple` via :func:`~tensorial.gcnn.atomic.graph_from_ase` the
    first time it is accessed, with the given keyword arguments, and the result is
    cached.

    Args:
        structures: already-loaded :class:`ase.Atoms` objects to serve
        as_graphs: keyword arguments for
            :func:`~tensorial.gcnn.atomic.graph_from_ase`, e.g. ``{"r_max": 5.0}``

    Example:
        >>> graphs = AseGraphs(structures, as_graphs={"r_max": 5.0})
        >>> graph = graphs[0]
    """

    def __init__(
        self,
        structures: "list[ase.Atoms]",
        as_graphs: dict[str, Any],
    ):
        # Params
        self._to_graphs: Final[dict[str, Any]] = as_graphs

        # State
        self._structures = structures

        if len(self) > 0:
            # Check that the parameters they passed are OK by requesting the first structure
            # to be converted
            self[0]  # noqa, pylint: disable=pointless-statement

    def __len__(self) -> int:
        return len(self._structures)

    def __getitem__(self, item: int) -> jraph.GraphsTuple:
        entry = self._structures[item]
        if not isinstance(entry, jraph.GraphsTuple):
            # Lazily convert the first time
            entry = atomic.graph_from_ase(entry, **self._to_graphs)
            self._structures[item] = entry

        return entry


class AseDataFetcher(data.DataFetcher[Sequence[jraph.GraphsTuple]]):
    """Fetch ASE structures from file(s) and convert them to graphs.

    Reads one or more files containing ASE :class:`ase.Atoms` objects (e.g. CIF, XYZ,
    extxyz) with :func:`ase.io.read`, and serves them as a single :class:`AseGraphs`
    sequence from :meth:`fetch`.  Use :class:`AseDataFetchers` to fetch several named
    datasets.

    Args:
        path: a path, or sequence of paths, to files containing ASE structures
        as_graphs: keyword arguments for
            :func:`~tensorial.gcnn.atomic.graph_from_ase`, e.g. ``{"r_max": 5.0}``
        limit: the maximum number of structures to read from each file
        read_kwargs: keyword arguments passed to :func:`ase.io.read`

    Example:
        >>> fetcher = AseDataFetcher("structures.xyz", as_graphs={"r_max": 5.0})
        >>> graphs = fetcher.fetch()
    """

    def __init__(
        self,
        path: PathSpec,
        as_graphs: dict[str, Any],
        limit: int | None = None,
        read_kwargs: dict[str, Any] | None = None,
    ):
        # Params
        self._filepath: Final[tuple[str]] = (path,) if isinstance(path, str) else tuple(path)
        self._as_graphs: Final[dict[str, Any]] = as_graphs
        self._read_kwargs: Final[dict[str, Any]] = _init_kwargs(limit, read_kwargs)

    def fetch(self) -> Sequence[jraph.GraphsTuple] | dict[str, Sequence[jraph.GraphsTuple]]:
        return self._make_sequence(self._read(self._filepath))

    def _make_sequence(self, structures: "list[ase.Atoms]") -> AseGraphs:
        return AseGraphs(structures, as_graphs=self._as_graphs)

    def _read(self, path: PathSpec) -> "list[ase.Atoms]":
        """Read ASE structure(s) from one or more files"""
        ase = utils.optional_import("ase")
        ase_io = utils.optional_import("ase.io")

        to_load: Final[tuple[str]] = (path,) if isinstance(path, str) else tuple(path)
        loaded: list[ase.Atoms] = []
        for entry in to_load:
            try:
                loaded.extend(ase_io.read(entry, **self._read_kwargs))
            except FileNotFoundError:
                raise ValueError(
                    f"Could not load ASE structures, the passed path does not exist: {entry}"
                ) from None

        return loaded


class AseDataFetchers(data.DataFetcher[dict[str, Sequence[jraph.GraphsTuple]]]):
    """Fetch several named ASE datasets and convert them to graphs.

    Wraps a mapping of name to path (or sequence of paths), fetching each named
    dataset via :class:`AseDataFetcher`.  :meth:`fetch` returns a dict mapping each
    name to its :class:`AseGraphs` sequence.

    Args:
        paths: mapping of name to a path, or sequence of paths, to files containing
            ASE structures
        as_graphs: keyword arguments for
            :func:`~tensorial.gcnn.atomic.graph_from_ase`, e.g. ``{"r_max": 5.0}``
        limit: the maximum number of structures to read from each file
        read_kwargs: keyword arguments passed to :func:`ase.io.read`

    Example:
        >>> fetchers = AseDataFetchers(
        ...     {"train": "train.xyz", "test": "test.xyz"}, as_graphs={"r_max": 5.0}
        ... )
        >>> datasets = fetchers.fetch()
    """

    def __init__(
        self,
        paths: dict[str, PathSpec],
        as_graphs: dict[str, Any],
        limit: int | None = None,
        read_kwargs: dict[str, Any] | None = None,
    ):
        # Params
        self._filepaths: Final[dict[str, PathSpec]] = paths
        self._as_graphs: Final[dict[str, Any]] = as_graphs
        self._read_kwargs: Final[dict[str, Any]] = _init_kwargs(limit, read_kwargs)

    def fetch(self) -> dict[str, Sequence[jraph.GraphsTuple]]:
        # ``None`` paths are legal (a split the user did not provide); skip them so
        # the resulting mapping may be a proper subset of ``self._filepaths``.
        return {
            name: AseDataFetcher(path=path, as_graphs=self._as_graphs).fetch()
            for name, path in self._filepaths.items()
            if path is not None
        }


class AseDataLoader(Sequence[jraph.GraphsTuple]):
    """Load ASE structures from file(s) and, optionally, convert them to graphs.

    Reads one or more files containing ASE :class:`ase.Atoms` objects (e.g. CIF, XYZ,
    extxyz) using :func:`ase.io.read`. If ``as_graphs`` is supplied, each structure is
    lazily converted to a :class:`jraph.GraphsTuple` via
    :func:`~tensorial.gcnn.atomic.graph_from_ase` the first time it is accessed, with the
    given keyword arguments.

    Args:
        path: a path, or sequence of paths, to files containing ASE structures
        limit: the maximum number of structures to read from each file
        read_kwargs: keyword arguments passed to :func:`ase.io.read`
        as_graphs: keyword arguments for :func:`~tensorial.gcnn.atomic.graph_from_ase`,
            e.g. ``{"r_max": 5.0}``. If ``None``, structures are returned as
            :class:`ase.Atoms`

    Example:
        >>> loader = AseDataLoader("structures.xyz", as_graphs={"r_max": 5.0})
        >>> graph = loader[0]
    """

    def __init__(
        self,
        path: str | Sequence[str],
        limit: int | None = None,
        read_kwargs: dict[str, Any] | None = None,
        as_graphs: dict[str, Any] | None = None,
    ):
        ase = utils.optional_import("ase")
        ase_io = utils.optional_import("ase.io")

        # Params
        self._filepath: Final[tuple[str]] = (path,) if isinstance(path, str) else tuple(path)
        self._limit: Final[int | None] = limit
        self._to_graphs: Final[dict[str, Any]] = as_graphs
        self._read_kwargs: Final[dict[str, Any]] = self._init_kwargs(limit, read_kwargs)

        try:
            loaded: "list[ase.Atoms]" = []
            for entry in self._filepath:
                loaded.extend(ase_io.read(entry, **self._read_kwargs))
            self._data: "list[ase.Atoms | jraph.GraphsTuple]" = (
                [loaded] if isinstance(loaded, ase.Atoms) else loaded
            )
        except FileNotFoundError:
            raise ValueError(
                f"Could not load ASE structures, the passed path does not exist: {path}"
            ) from None

        if self._to_graphs and len(self) > 0:
            # Check that the parameters they passed are OK by requesting the first structure
            # to be converted
            self[0]  # noqa, pylint: disable=pointless-statement

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: int) -> Any:
        entry = self._data[item]
        if self._to_graphs and not isinstance(entry, jraph.GraphsTuple):
            # Lazily convert the first time
            entry = atomic.graph_from_ase(entry, **self._to_graphs)
            self._data[item] = entry
        return entry

    @staticmethod
    def _init_kwargs(
        limit: int | None, read_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if read_kwargs is None:
            read_kwargs = {"index": f":{limit}" if limit is not None else ":"}
        else:
            if limit is not None:
                read_kwargs["index"] = f":{limit}"
            elif "index" not in read_kwargs:
                read_kwargs["index"] = ":"
        return read_kwargs


def ase_graph_module_random_split(
    path,
    as_graphs: dict[str, Any],
    train_val_test_split: Sequence[float] = (0.85, 0.05, 0.1),
    *,
    batch_size: int = 32,
    batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
    pad_to_multiple: "int | str | jax.Device | None" = None,
) -> "gcnn.data.GraphDataModule[gcnn.data.GraphDataset]":
    """Build a :class:`~tensorial.gcnn.data.GraphDataModule` for random splitting.

    Loads ASE structures from ``path`` (via :class:`AseDataFetcher`) and splits them
    into train/val/test at random.

    Args:
        path: a path, or sequence of paths, to files containing ASE structures
        as_graphs: keyword arguments for
            :func:`~tensorial.gcnn.atomic.graph_from_ase`, e.g. ``{"r_max": 5.0}``
        train_val_test_split: the (train, val, test) fractions to use
    """
    return _datamodule.GraphDataModule.from_random_split(
        AseDataFetcher(path, as_graphs=as_graphs),
        train_val_test_split,
        batch_size=batch_size,
        batch_mode=batch_mode,
        pad_to_multiple=pad_to_multiple,
    )


def ase_graph_module_kfold(
    path,
    as_graphs: dict[str, Any],
    fold: int,
    n_folds: int = 5,
    *,
    seed: int | None = 42,
    test_fraction: float = 0.1,
    batch_size: int = 32,
    batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
    pad_to_multiple: "int | str | jax.Device | None" = None,
) -> "gcnn.data.GraphDataModule[gcnn.data.GraphDataset]":
    """Build a :class:`~tensorial.gcnn.data.GraphDataModule` for k-fold cross-validation.

    Loads ASE structures from ``path`` (via :class:`AseDataFetcher`) and holds out one
    fold as the validation set.  The test set is carved out first using
    ``test_fraction``; only that fraction is configurable here, while the
    train/val split is driven by ``n_folds`` and ``fold``.

    Args:
        path: a path, or sequence of paths, to files containing ASE structures
        as_graphs: keyword arguments for
            :func:`~tensorial.gcnn.atomic.graph_from_ase`, e.g. ``{"r_max": 5.0}``
        fold: the fold index to hold out for validation
        n_folds: the total number of folds
        seed: the random seed
        test_fraction: the fraction of data to carve out as the test set
    """
    return _datamodule.GraphDataModule.from_kfold(
        AseDataFetcher(path, as_graphs=as_graphs),
        fold=fold,
        n_folds=n_folds,
        seed=seed,
        test_fraction=test_fraction,
        batch_size=batch_size,
        batch_mode=batch_mode,
        pad_to_multiple=pad_to_multiple,
    )


def ase_graph_module_from_datasets(
    # pylint: disable=arguments-differ
    as_graphs: dict[str, Any],
    *,
    train=None,
    val=None,
    test=None,
    batch_size: int = 32,
    batch_mode: "gcnn.data.BatchMode | str" = _common.BatchMode.IMPLICIT,
    pad_to_multiple: "int | str | jax.Device | None" = None,
) -> "gcnn.data.GraphDataModule[gcnn.data.PreSplitDataset]":
    """Build a :class:`~tensorial.gcnn.data.GraphDataModule` from already-split datasets.

    Loads ASE structures from the ``train``, ``val`` and ``test`` paths (via
    :class:`AseDataFetchers`) and serves them as the corresponding splits.

    Args:
        as_graphs: keyword arguments for
            :func:`~tensorial.gcnn.atomic.graph_from_ase`, e.g. ``{"r_max": 5.0}``
        train: path(s) to the training structures, or ``None``
        val: path(s) to the validation structures, or ``None``
        test: path(s) to the test structures, or ``None``
    """
    return _datamodule.GraphDataModule.from_datasets(
        AseDataFetchers({"train": train, "val": val, "test": test}, as_graphs=as_graphs),
        batch_size=batch_size,
        batch_mode=batch_mode,
        pad_to_multiple=pad_to_multiple,
    )
