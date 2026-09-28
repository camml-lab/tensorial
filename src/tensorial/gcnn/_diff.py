import abc
from collections.abc import Sequence
import dataclasses
import logging
import re
from typing import TYPE_CHECKING, Final, Protocol, Union

import flax.core
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph

from . import _base, _tree
from .. import base

if TYPE_CHECKING:
    from tensorial import gcnn

__all__ = ("diff",)

_LOGGER = logging.getLogger(__name__)
DERIV_DELIMITER: Final[str] = ","
ArgumentSpecifier = Union[int, "gcnn.typing.TreePath"]

#: Differentiation modes accepted by :func:`diff`.
MODES: Final[tuple[str, ...]] = ("rev", "fwd", "auto")
#: Assumed extent of an index that is not the leading (graph-extensive) one.  Only used by
#: ``mode="auto"`` to compare input and output sizes; the comparison is decided by the
#: extensive factor, so this value only matters for near-ties.
CARTESIAN_DIM: Final[int] = 3


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got: {mode!r}")
    return mode


class DerivableGraphFunction(Protocol):
    def __call__(
        self, graph: jraph.GraphsTuple, *args: jt.PyTree
    ) -> jt.Array | tuple[jt.Array, jraph.GraphsTuple]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class GraphEntrySpec:
    """
    A specification that identifies a particular entry in a hierarchical data structure
    (e.g., a PyTree) along with optional index labels used for differentiable computations.

    Attributes:
        key_path (Optional[gcnn.typing.TreePath]): A path to the target node in the PyTree,
            typically represented as a tuple of keys (e.g., strings or integers).
        indices (Optional[str]): A string representing symbolic indices, often used to
            annotate tensor dimensions for operations like differentiation.
    """

    key_path: ArgumentSpecifier | None
    indices: str | None

    @classmethod
    def create(cls, spec: "GraphEntrySpecLike") -> "GraphEntrySpec":
        if isinstance(spec, GraphEntrySpec):
            return spec
        if spec is None:
            return GraphEntrySpec(None, None)

        match = re.match(r"^(?:(.*?))?(?::(.*))?$", spec)
        if not match:
            raise ValueError(f"Could not parse the expression: {spec}")

        groups = match.groups()
        try:
            kee_path = int(groups[0])
        except ValueError:
            kee_path = _tree.path_from_str(groups[0])

        return GraphEntrySpec(kee_path, groups[1])

    @property
    def safe_indices(self) -> str:
        return self.indices if self.indices is not None else ""

    @property
    def rank(self) -> int | None:
        return len(self.indices) if self.indices is not None else None

    def __str__(self) -> str:
        rep = []
        if self.key_path is not None:
            key_path = (
                str(self.key_path)
                if isinstance(self.key_path, int)
                else _tree.path_to_str(self.key_path)
            )
            rep.append(key_path)
        if self.indices is not None:
            rep.append(f":{self.indices}")
        return "".join(rep)

    def __repr__(self) -> str:
        return f"GraphEntrySpec('{self.__str__()}')"

    def __truediv__(self, other: "GraphEntrySpecLike") -> "SingleDerivative":
        return SingleDerivative.create(self, other)

    def index_union(self, other: "GraphEntrySpec") -> str | None:
        # use dictionary keys as a set
        out = dict() if not self.indices else dict.fromkeys(self.indices)
        if other.indices:
            out |= dict.fromkeys(other.indices)
        if not out:
            return None

        return "".join(out)

    def indices_intersection(self, other: "GraphEntrySpec") -> str:
        if not self.indices or not other.indices:
            return ""

        out = [index for index in self.indices if index in other.indices]
        return "".join(out)


GraphEntrySpecLike = str | GraphEntrySpec


class Derivative(abc.ABC):
    @property
    @abc.abstractmethod
    def of(self) -> GraphEntrySpec:
        """Derivative of"""

    @property
    @abc.abstractmethod
    def graph_tuple_paths(self) -> "dict[gcnn.typing.TreePath, int]": ...

    @property
    @abc.abstractmethod
    def argnum_paths(self) -> dict[int, int]: ...

    @property
    @abc.abstractmethod
    def out(self) -> GraphEntrySpec:
        """Derivative output"""

    def evaluator(
        self,
        func: DerivableGraphFunction,
        return_graph: bool,
        argnum: int = 0,
        scale: float = 1.0,
        at: dict | None = None,
        mode: str = "rev",
    ) -> "Evaluator":
        if at is not None:
            at = flax.core.FrozenDict(at)
        return Evaluator(func, self, return_graph, argnum, scale=scale, at=at, mode=mode)

    @abc.abstractmethod
    def build_derivative_fn(
        self,
        func: DerivableGraphFunction,
        return_graph: bool,
        argnum: int,
        mode: str = "rev",
        extents: "dict[str, int] | None" = None,
    ) -> DerivableGraphFunction:
        """Get evaluate function from derivative"""

    @abc.abstractmethod
    def index_extents(
        self, graph: jraph.GraphsTuple, args: "Sequence[jt.PyTree]"
    ) -> dict[str, int]:
        """Map each index label to its extent, for ``mode="auto"``.

        The ``wrt`` values are concrete by the time this is called, so the labels they carry
        are exact.  For a chain this is what lets a later link size an intermediate whose
        shape is not otherwise knowable without tracing the model.
        """

    def adapt(self, func: "gcnn.typing.ExGraphFunction") -> DerivableGraphFunction:
        return _base.adapt(
            func,
            *self.graph_tuple_paths,
            outs=tuple() if not self.of or not self.of.key_path else [self.of.key_path],
            return_graphs=True,
        )


def infer_out_indices(of: GraphEntrySpec, wrt: GraphEntrySpec) -> str:
    # All indices resulting from differentiating 'of' with respect to 'wrt'
    all_deriv_indices = of.safe_indices + wrt.safe_indices

    # Find indices common to both 'of' and 'wrt' — these will be reduced (summed over)
    shared_indices = of.indices_intersection(wrt)
    of_reduce = tuple(of.indices.index(i) for i in shared_indices)

    return "".join([idx for i, idx in enumerate(all_deriv_indices) if i not in of_reduce])


@dataclasses.dataclass(frozen=True, slots=True)
class SingleDerivative(Derivative):
    _of: GraphEntrySpec
    _wrt: GraphEntrySpec
    _out: GraphEntrySpec

    # will be set in __post_init__
    _pre_reduce: tuple[int, ...] = dataclasses.field(init=False)
    _post_reduce: tuple[int, ...] = dataclasses.field(init=False)
    _post_permute: tuple[int, ...] = dataclasses.field(init=False)
    _actual_out: GraphEntrySpec = dataclasses.field(init=False)

    def __post_init__(self):
        # All indices resulting from differentiating 'of' with respect to 'wrt'
        all_deriv_indices = self.of.safe_indices + self.wrt.safe_indices

        # now check that the output indices are some subset of these
        if self._out.indices is not None and not set(self._out.indices).issubset(all_deriv_indices):
            raise ValueError(
                f"The passed output indices {self._out.indices} include some that are "
                f"not in of ({self.of.indices}) or wrt ({self.wrt.indices})"
            )

        # Find indices common to both 'of' and 'wrt' — these will be reduced (summed over)
        unreduced_indices = []
        pre_reduce = []
        if self.of.indices is not None:
            for i, index in enumerate(self.of.indices):
                index_not_in_out = self._out.indices is not None and index not in self._out.indices
                index_in_wrt = self.wrt.indices is not None and index in self.wrt.indices
                if index_not_in_out or index_in_wrt:
                    pre_reduce.append(i)
                else:
                    unreduced_indices.append(index)
        object.__setattr__(self, "_pre_reduce", tuple(pre_reduce))

        num_after_pre_reduce = len(unreduced_indices)
        post_reduce = []
        if self.wrt.indices is not None:
            for i, index in enumerate(self.wrt.indices):
                if self._out.indices is not None and index not in self._out.indices:
                    post_reduce.append(i + num_after_pre_reduce)
                else:
                    unreduced_indices.append(index)
        object.__setattr__(self, "_post_reduce", tuple(post_reduce))

        # Map output indices to their new position in the output tensor
        post_permute = (
            find_index_permutation(unreduced_indices, self._out.indices)
            if self._out.indices is not None
            else []
        )
        object.__setattr__(self, "_post_permute", tuple(post_permute))

        if self._out.indices is None:
            out_indices = "".join(unreduced_indices)
            if post_permute:
                out_indices = "".join(out_indices for i in post_permute)

            out = GraphEntrySpec(self._out.key_path, out_indices)
        else:
            out = self._out

        object.__setattr__(self, "_actual_out", out)

    @classmethod
    def create(
        cls,
        of: GraphEntrySpecLike,
        wrt: GraphEntrySpecLike,
        out: GraphEntrySpecLike | None = None,
    ) -> "SingleDerivative":
        of = GraphEntrySpec.create(of)
        wrt = GraphEntrySpec.create(wrt)
        out = GraphEntrySpec.create(out)
        return SingleDerivative(of, wrt, out)

    @property
    def of(self) -> GraphEntrySpec:
        """The entry being differentiated."""
        return self._of

    @property
    def wrt(self) -> GraphEntrySpec:
        """The entry with respect to which the derivative is taken."""
        return self._wrt

    @property
    def graph_tuple_paths(self) -> "dict[gcnn.typing.TreePath, int]":
        if isinstance(self._wrt.key_path, int):
            return {}

        return {self._wrt.key_path: 0}

    @property
    def argnum_paths(self) -> dict[int, int]:
        if not isinstance(self._wrt.key_path, int):
            return {}

        return {self._wrt.key_path: 0}

    @property
    def out(self) -> GraphEntrySpec:
        """The entry containing the derivative output."""
        return self._actual_out

    def __str__(self) -> str:
        return f"∂{self.of}/∂{self.wrt}->{self.out}"

    def __truediv__(self, other: "GraphEntrySpecLike | SingleDerivative") -> "MultiDerivative":
        if isinstance(other, SingleDerivative):
            return MultiDerivative((self, other))

        wrt = GraphEntrySpec.create(other)
        return MultiDerivative((self, SingleDerivative.create(self.out, wrt)))

    def broadcast_wrt(self, graph: jraph.GraphsTuple, value: jt.Array) -> jt.Array:
        """Give ``value`` its leading graph dimension if it was passed without one."""
        if (
            self.wrt.rank is not None
            and not isinstance(value, float)
            and value.ndim == self.wrt.rank - 1
        ):
            return jnp.broadcast_to(
                value, (_tree.num_entries(self.wrt.key_path[0], graph), *value.shape)
            )
        return value

    def index_extents(
        self, graph: jraph.GraphsTuple, args: "Sequence[jt.PyTree]"
    ) -> dict[str, int]:
        extents: dict[str, int] = {}
        if self.wrt.indices and args:
            value = base.as_array(self.broadcast_wrt(graph, args[0]))
            for i, label in enumerate(self.wrt.indices):
                if i < len(value.shape):
                    extents[label] = value.shape[i]
        if self.of.indices and self.of.key_path and not isinstance(self.of.key_path, int):
            # The leading index of a graph entry counts its nodes / edges / graphs
            extents.setdefault(self.of.indices[0], _tree.num_entries(self.of.key_path[0], graph))
        return extents

    def differentiated_size(
        self, graph: jraph.GraphsTuple, extents: "dict[str, int] | None" = None
    ) -> int:
        """Number of scalars in the value that gets differentiated.

        This is the size of ``of`` *after* :meth:`_pre_process` has summed away the indices
        that do not survive into the output, which is what sets the cost of reverse mode:
        one backward pass per output.

        The leading index of a graph entry is exact (it is the node/edge/graph count); any
        other surviving index is assumed to be :data:`CARTESIAN_DIM`.  Only the extensive
        factor decides the comparison in practice, so the assumption is not load bearing.
        """
        if self.of.indices is None:
            return 1

        extents = extents or {}
        size = 1
        for i, label in enumerate(self.of.indices):
            if i in self._pre_reduce:
                continue  # summed away before the derivative is taken
            if label in extents:
                size *= extents[label]
            elif i == 0 and self.of.key_path and not isinstance(self.of.key_path, int):
                size *= _tree.num_entries(self.of.key_path[0], graph)
            else:
                size *= CARTESIAN_DIM
        return size

    def choose_mode(
        self,
        graph: jraph.GraphsTuple,
        wrt_value: jt.Array,
        extents: "dict[str, int] | None" = None,
    ) -> str:
        """Pick forward or reverse mode from the shapes of this derivative.

        Reverse mode costs one pass per *output* of the differentiated function, forward mode
        one pass per *input*, so the cheaper one is decided by which side is smaller.  Ties go
        to forward mode, which needs no tape and so has the smaller memory footprint.
        """
        n_out = self.differentiated_size(graph, extents)
        n_in = base.as_array(wrt_value).size
        mode = "rev" if n_out < n_in else "fwd"
        _LOGGER.debug("%s: choosing %s mode (%i outputs vs %i inputs)", self, mode, n_out, n_in)
        return mode

    def build_derivative_fn(
        self,
        func: DerivableGraphFunction,
        return_graph: bool,
        argnum: int,
        mode: str = "rev",
        extents: "dict[str, int] | None" = None,
    ) -> DerivableGraphFunction:
        if not argnum >= 0:
            raise ValueError(f"argnum must be >= 0, got: {argnum}")
        _check_mode(mode)

        def _diff_and_pre_process(
            graph: jraph.GraphsTuple, *args: jt.PyTree
        ) -> tuple[jt.Array, jraph.GraphsTuple]:
            value, graph = func(graph, *args)
            value, graph = self._pre_process(value, graph)

            return value, graph

        built: dict[str, DerivableGraphFunction] = {}

        def do_diff(mode_: str) -> DerivableGraphFunction:
            """Build (and cache) the jax transform for one mode."""
            if mode_ not in built:
                if not self.out.indices:
                    # Scalar valued: a single VJP gives the whole gradient, mode is irrelevant
                    diff_fn = jax.grad
                else:
                    diff_fn = jax.jacfwd if mode_ == "fwd" else jax.jacrev
                built[mode_] = diff_fn(_diff_and_pre_process, argnums=1 + argnum, has_aux=True)
            return built[mode_]

        def _diff_fn(
            graph: jraph.GraphsTuple, *args: jt.PyTree
        ) -> jt.Array | tuple[jt.Array, jraph.GraphsTuple]:
            if len(args) <= argnum:
                raise ValueError(
                    f"Derivative needs to be taken wrt argument {argnum}, "
                    f"but only {len(args)} were passed.  Did you forget to pass a value for "
                    f"the value at which you would like the derivative to be evaluated?"
                )

            value = args[argnum]
            broadcast = self.broadcast_wrt(graph, value)
            if broadcast is not value:
                # Broadcast to correct leading dimension based on the graph's number of entries
                value = broadcast
                args = (*args[:argnum], value, *args[argnum + 1 :])

            self._check_shape("wrt", self.wrt, value)

            chosen = self.choose_mode(graph, value, extents) if mode == "auto" else mode
            value, graph = do_diff(chosen)(graph, *args)
            value, graph = self._post_process(value, graph)

            if return_graph:
                return value, graph

            return value

        return _diff_fn

    def _pre_process(
        self, value: jt.Array, graph: jraph.GraphsTuple
    ) -> tuple[jt.Array, jraph.GraphsTuple]:
        self._check_shape("of", self.of, value)

        if self._pre_reduce:
            value = base.as_array(value).sum(axis=self._pre_reduce)

        return value, graph

    def _post_process(
        self, value: jt.Array, graph: jraph.GraphsTuple
    ) -> tuple[jt.Array, jraph.GraphsTuple]:
        if self._post_reduce:
            value = base.as_array(value).sum(axis=self._post_reduce)
        if self._post_permute:
            value = value.transpose(self._post_permute)
        self._check_shape("out", self.out, value)

        return value, graph

    @staticmethod
    def _check_shape(stage: str, spec: GraphEntrySpec, value: jt.Array):
        if spec.indices is None:
            return  # Nothing to check

        value = base.as_array(value)
        if len(spec.indices) != len(value.shape):
            raise ValueError(
                f"The passed '{stage}' indices `{spec.indices}` do not match the rank of the "
                f"value shape: {value.shape}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MultiDerivative(Derivative):
    parts: tuple[SingleDerivative, ...]

    @classmethod
    def create(
        cls,
        of: GraphEntrySpecLike,
        wrt: str | Sequence[GraphEntrySpecLike],
        out: GraphEntrySpecLike | None = None,
    ) -> "MultiDerivative":
        if isinstance(wrt, str):
            wrt = wrt.split(",")

        if len(wrt) == 1:
            return MultiDerivative((SingleDerivative.create(of, wrt[0], out),))

        # First
        parts = [SingleDerivative.create(of, wrt[0])]
        # Middle
        for part in wrt[1:-1]:
            parts.append(SingleDerivative.create(parts[-1].out, part))
        # Last
        parts.append(SingleDerivative.create(parts[-1].out, wrt[-1], out))

        return MultiDerivative(tuple(parts))

    @property
    def of(self) -> GraphEntrySpec:
        return self.parts[0].of

    @property
    def graph_tuple_paths(self) -> "dict[gcnn.typing.TreePath, int]":
        paths: "dict[gcnn.typing.TreePath, int]" = {}
        wrt_map = []
        for part in self.parts:
            if not isinstance(part.wrt.key_path, int):
                wrt_map.append(paths.setdefault(part.wrt.key_path, len(paths)))
        return paths

    @property
    def argnum_paths(self) -> dict[int, int]:
        start_idx = len(self.graph_tuple_paths)
        paths = {}
        for part in self.parts:
            if isinstance(part.wrt.key_path, int):
                paths.setdefault(part.wrt.key_path, start_idx + len(paths))

        return paths

    @property
    def out(self) -> GraphEntrySpec:
        return self.parts[-1].out

    @property
    def paths(self) -> dict[ArgumentSpecifier, int]:
        paths = {}
        wrt_map = []
        for part in self.parts:
            wrt_map.append(paths.setdefault(part.wrt.key_path, len(paths)))
        return paths

    def __str__(self) -> str:
        parts = [f"∂{self.of}/∂{self[0].wrt}"]

        for deriv in self[1:]:
            parts.append(f"∂{deriv.wrt}")

        parts.append(f"->{self.out}")
        return "".join(parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __getitem__(self, item) -> SingleDerivative | tuple[SingleDerivative, ...]:
        return self.parts[item]

    def __iter__(self):
        yield from self.parts.__iter__()

    def build_derivative_fn(
        self,
        func: DerivableGraphFunction,
        return_graph: bool,
        argnum: int,
        mode: str = "rev",
        extents: "dict[str, int] | None" = None,
    ) -> DerivableGraphFunction:
        # Each link gets the same `mode`, but "auto" is resolved per link at call time, so a
        # chain whose levels want different modes will get them.
        _check_mode(mode)

        argnums = self._part_argnums()

        func = self[0].build_derivative_fn(
            func, return_graph=return_graph, argnum=argnums[0], mode=mode, extents=extents
        )
        for part, argnum_ in zip(self[1:], argnums[1:]):
            func = part.build_derivative_fn(
                func, return_graph=return_graph, argnum=argnum_, mode=mode, extents=extents
            )

        return func

    def _part_argnums(self) -> list[int]:
        """Position in the argument tuple that each link differentiates with respect to."""
        argnums = []
        for part in self:
            wrt_path = part.wrt.key_path
            if isinstance(wrt_path, int):
                argnums.append(self.argnum_paths[wrt_path])
            else:
                argnums.append(self.graph_tuple_paths[wrt_path])
        return argnums

    def index_extents(
        self, graph: jraph.GraphsTuple, args: "Sequence[jt.PyTree]"
    ) -> dict[str, int]:
        """Merge the extents every link can see.

        Intermediate ``of`` specs carry no key path, so their labels can only be sized from
        the links that introduced them -- which is exactly what this merge provides.
        """
        extents: dict[str, int] = {}
        for part, argnum in zip(self, self._part_argnums()):
            if argnum < len(args):
                extents.update(part.index_extents(graph, (args[argnum],)))
        return extents


def process_paths(parts: Sequence[SingleDerivative]):
    """Process paths from a sequence of single derivatives.

    Args:
        parts: A sequence of SingleDerivative objects

    Returns:
        A tuple of (paths, wrt_map, arg_parts) where paths maps tree paths to indices,
        wrt_map contains the wrt path indices, and arg_parts contains argument indices.
    """
    paths: dict[gcnn.typing.TreePath, int] = {}
    wrt_map: list[int] = []
    arg_parts = []
    for part in parts:
        if isinstance(part.wrt.key_path, int):
            arg_parts.append(part.wrt.key_path)
        else:
            wrt_map.append(paths.setdefault(part.wrt.key_path, len(paths)))

    for part in parts:
        pass

    return paths


@dataclasses.dataclass(frozen=True)
class Evaluator:
    func: DerivableGraphFunction
    spec: Derivative
    return_graph: bool
    argnum: int
    scale: float = 1.0
    at: flax.core.FrozenDict[str, jt.PyTree] | None = dataclasses.field(default=None, hash=False)
    mode: str = "rev"

    # will be set in __post_init__
    _evaluate_at: DerivableGraphFunction = dataclasses.field(init=False)
    # Shared with the built closures; refreshed on every call so that "auto" sees the
    # current graph's dimensions rather than whatever the last call had.
    _extents: dict[str, int] = dataclasses.field(init=False, default_factory=dict, hash=False)

    def __post_init__(self):
        _check_mode(self.mode)
        extents: dict[str, int] = {}
        object.__setattr__(self, "_extents", extents)
        object.__setattr__(
            self,
            "_evaluate_at",
            self.spec.build_derivative_fn(
                self.func,
                return_graph=True,
                argnum=self.argnum,
                mode=self.mode,
                extents=extents,
            ),
        )

    def __call__(
        self, graph: jraph.GraphsTuple, *args, **kwargs: dict[str, jt.PyTree]
    ) -> jt.Array | tuple[jt.Array, jraph.GraphsTuple]:
        values: dict[int, jt.PyTree] = {}

        for name, value in (self.at or {}).items():
            tree_path = _tree.path_from_str(name)
            idx: int = self.spec.graph_tuple_paths[tree_path]
            values[idx] = value

        # kwargs take precedence, so come second
        for name, value in kwargs.items():
            idx: int = self.spec.graph_tuple_paths[_tree.path_from_str(name)]
            values[idx] = value

        diff_args = tuple(value for _, value in sorted(values.items())) + args
        if self.mode == "auto":
            self._extents.clear()
            self._extents.update(self.spec.index_extents(graph, diff_args))
        value, graph_out = self._evaluate_at(graph, *diff_args)
        if self.spec.out.indices is not None and not len(value.shape) == len(self.spec.out.indices):
            raise ValueError(
                f"The output array rank ({len(value.shape)}) does not match the "
                f"passed number of output indices '{self.spec.out.indices}' "
                f"({len(self.spec.out.indices)})"
            )
        value = self.scale * value

        if self.return_graph:
            return value, graph_out

        return value


def diff(
    *func_of,
    wrt: GraphEntrySpecLike | Sequence[GraphEntrySpecLike],
    of: GraphEntrySpecLike | None = None,
    out: GraphEntrySpecLike = None,
    scale: float = 1.0,
    at: dict | None = None,
    return_graph=False,
    mode: str = "rev",
) -> Evaluator:
    """
    Constructs a JAX-compatible evaluator for computing single or multiple derivatives
    of a scalar function (e.g., energy) defined over a Graph.

    This function acts as a factory, routing the request to create either a
    SingleDerivative or MultiDerivative object based on the `wrt` argument.

    The key feature of this function is the use of string-based tensor index notation
    (e.g., 'nodes.positions:Iγ') for specifying differentiation targets and output shape.

    Args:
        *func_of: Either ``(func)``, where ``func`` is the energy function, or
            ``(func, of)``, where ``of`` is the scalar entry (e.g.,
            ``globals.energy``) to differentiate. ``func`` is the function
            (Graph -> Graph) whose output is differentiated, and ``of`` (optional)
            specifies the scalar entry within the output graph of ``func`` to
            differentiate, defaulting to the sole scalar output if omitted.
        wrt: The input entries of the graph with respect to which the derivative is
            taken. This must be a string or a sequence of strings, specifying the
            index notation for the input: for example, ``nodes.positions:Iγ`` means
            differentiate w.r.t. the ``γ`` component of the position of node ``I``.
        out: The index notation for the desired output tensor shape. This string
            defines the contraction of the indices from ``wrt``. For example, if
            ``wrt=['field:α', 'field:β', 'positions:Iγ']``, then
            ``out=':Iγαβ'`` specifies a rank-4 tensor output with indices
            ``I, γ, α, β``. If omitted, the indices are concatenated in the order
            they appear in ``wrt``.
        scale: A scalar factor to multiply the final derivative result by. Defaults
            to 1.0.
        at: A dictionary mapping GraphEntrySpecLike strings (without indices) to
            jax.numpy arrays, specifying the value at which to evaluate the derivative
            for those entries. These entries will be held constant, e.g.
            ``{'globals.electric_field': jnp.zeros(3)}``.
        return_graph: If True, the derivative tensor is packaged into a new Graph
            object under the name specified by the ``out`` argument. If False, the
            function returns the raw derivative tensor. Defaults to False.
        mode: "rev" (default) uses reverse-mode AD (jax.jacrev). "fwd" uses
            forward-mode AD (jax.jacfwd), which is more memory-efficient when
            dim(input) << dim(output).  "auto" picks per derivative from the shapes:
            reverse mode costs one pass per output and forward mode one pass per input,
            so whichever side is smaller wins, with ties going to forward mode.  In a
            chain (several entries in ``wrt``) each link chooses independently, which
            matters when the levels disagree -- a mixed second derivative such as
            d2E/dmu dB_ext wants reverse for the inner (scalar energy, many inputs) and
            forward for the outer (few inputs, many outputs).

    Returns:
        A callable object that takes a Graph and returns the computed derivative
        tensor (or a Graph containing it).

    Raises:
        TypeError: If the arguments do not conform to the expected types.

    Note:
        The index notation used in ``wrt`` and ``out`` must adhere to the library's
        conventions for Graph entry keys and indices. For multi-derivatives, the
        number of unique indices in ``out`` must match the number of indices in
        ``wrt``.
    """
    if len(func_of) == 1:
        func = func_of[0]
    else:
        if of is not None:
            raise ValueError(
                "You need to call diff() as either diff(fn, 'globals.energy', ...) or "
                "diff(fn, of='globals.energy', ...) but you cannot provide both 2 positional "
                "arguments and `of`"
            )
        func, of = func_of

    _check_mode(mode)

    if isinstance(wrt, str):
        deriv = SingleDerivative.create(of, wrt, out)
    elif len(wrt) == 1:
        deriv = SingleDerivative.create(of, wrt[0], out)
    else:
        deriv = MultiDerivative.create(of, wrt, out)

    return deriv.evaluator(deriv.adapt(func), return_graph, scale=scale, at=at, mode=mode)


def ordered_unique_indices(lst):
    """Return indices of unique elements in order of first appearance."""
    seen = {}
    return [i for i, x in enumerate(lst) if x not in seen and not seen.setdefault(x, True)]


def find_index_permutation(lst_a, lst_b) -> list[int]:
    """Find the permutation mapping elements from lst_a to their positions in lst_b.

    Args:
        lst_a: Source list of unique values.
        lst_b: Target list containing all values from lst_a (possibly reordered).

    Returns:
        A list of indices such that lst_a[i] == lst_b[result[i]] for all i.
    """
    index_map = {value: idx for idx, value in enumerate(lst_a)}
    return [index_map[x] for x in lst_b]
