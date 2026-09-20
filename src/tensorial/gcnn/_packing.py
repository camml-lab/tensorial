from typing import TYPE_CHECKING

import e3nn_jax as e3j
from flax import linen
import jax.numpy as jnp
import jraph
from pytray import tree

from . import _base, _tree, keys
from .. import base

if TYPE_CHECKING:
    import tensorial
    from tensorial import gcnn


__all__ = ("Pack",)

_COMPONENTS = ("nodes", "edges", "globals")


class Pack(linen.Module):
    """Pack attributes into a direct sum of irreps at ``out_field``.

    Each entry of ``attrs`` is a **full tree path** whose first element is
    ``nodes``, ``edges`` or ``globals``, so a single call can bring together
    attributes from anywhere in the graph. Likewise ``out_field`` is a full
    path of the form ``"nodes.attributes"``, ``"edges.attributes"`` or
    ``"globals.attributes"`` identifying the destination component and slot.

    The values are converted to a single :class:`e3nn_jax.IrrepsArray` with
    :func:`tensorial.create_tensor` and written back to the graph at
    ``out_field``. Values coming from ``globals`` are broadcast across the
    destination's leading axis (repeat by ``n_node`` / ``n_edge`` for batched
    graphs), so a per-graph quantity can be mixed with per-node or per-edge
    attributes in one embedding.

    Parameters
    ----------
    attrs : IrrepsTree
        Mapping ``{attr_path: Tensorial}``. Every key must start with one of
        ``nodes``, ``edges`` or ``globals`` (e.g. ``"nodes.species"``,
        ``"globals.temperature"``).
    out_field : str | tuple
        Full tree path where the resulting ``IrrepsArray`` is stored. The
        first key must be one of ``nodes``, ``edges`` or ``globals``.
    shape_from : str | None, optional
        Name of an existing attribute on the destination component used to
        derive a static length when broadcasting globals. If ``None``, the
        length is computed as ``jnp.sum(graph.n_node / n_edge)``, which can
        trigger recompilation.

    Examples
    --------
    Bring a per-node ``species`` together with a per-graph ``temperature``
    into ``graph.nodes.attributes`` — the global is broadcast across nodes::

        embedding = Embedding(
            attrs={
                "nodes.species":       tensorial.Attr("0e"),
                "globals.temperature": tensorial.Attr("0e"),
            },
            out_field="nodes.attributes",
        )
        graph = embedding(graph)

    Pack a single edge attribute into ``graph.edges.attributes``::

        embedding = Embedding(
            attrs={"edges.displacement": tensorial.Attr("1o")},
            out_field="edges.attributes",
        )
        graph = embedding(graph)

    Pack a global scalar into ``graph.globals.attributes``::

        embedding = Embedding(
            attrs={"globals.temperature": tensorial.Attr("0e")},
            out_field="globals.attributes",
        )
        graph = embedding(graph)
    """

    attrs: "tensorial.IrrepsTree"
    out_field: "gcnn.typing.TreePathLike"
    shape_from: str | None = keys.POSITIONS

    @_base.shape_check
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        dest_path = _tree.path_from_str(self.out_field)
        self._check_component(dest_path, self.out_field)
        dest_component = dest_path[0]

        if isinstance(self.attrs, (dict, linen.FrozenDict)):
            values = []
            for key, attr in self.attrs.items():
                src_path = _tree.path_from_str(key)
                self._check_component(src_path, key)

                raw = self._get_or_raise(graph, src_path)
                value = base.create_tensor(attr, raw)

                if src_path[0] == "globals" and dest_component != "globals":
                    value = self._broadcast_globals(value, graph, dest_component)

                values.append(value)

            packed = e3j.concatenate(values)
        else:
            # ``attrs`` describes a whole component at once (e.g. an IrrepsObj
            # type); pull it directly from the destination component.
            source_vals = getattr(graph, dest_component)
            packed = base.create_tensor(self.attrs, source_vals)

        graph_dict = graph._asdict()
        tree.set_by_path(graph_dict, dest_path, packed)
        return jraph.GraphsTuple(**graph_dict)

    # ------------------------------------------------------------------ #
    # helpers

    @staticmethod
    def _check_component(path: tuple, original) -> None:
        if not path or path[0] not in _COMPONENTS:
            raise ValueError(
                f"Path '{_tree.path_to_str(original)}' must start with one of "
                f"{', '.join(_COMPONENTS)}, got '{path[0] if path else ''}'"
            )

    @staticmethod
    def _get_or_raise(graph, path):
        try:
            return _tree.get(graph, path)
        except KeyError:
            root = path[0]
            available = _tree.get(graph, root)
            available_keys = available.keys() if hasattr(available, "keys") else ()
            raise ValueError(
                f"Did not find '{_tree.path_to_str(path[1:])}' in {root}, "
                f"available entries are: {', '.join(available_keys)}"
            ) from None

    def _broadcast_globals(
        self, value: e3j.IrrepsArray, graph: jraph.GraphsTuple, dest_component: str
    ) -> e3j.IrrepsArray:
        """Repeat a per-graph global along the destination's leading axis."""
        if dest_component == "nodes":
            if self.shape_from is not None and self.shape_from in graph.nodes:
                length = graph.nodes[self.shape_from].shape[0]
            else:
                length = jnp.sum(graph.n_node)
            n_per_graph = graph.n_node
        elif dest_component == "edges":
            length = jnp.sum(graph.n_edge)
            n_per_graph = graph.n_edge
        else:
            return value

        if len(value.shape) < 2:
            return value.broadcast_to((length, *value.shape))

        array = jnp.repeat(value.array, n_per_graph, axis=0, total_repeat_length=length)
        return e3j.IrrepsArray(value.irreps, array)
