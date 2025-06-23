from collections.abc import Callable
from typing import Optional, Union

import beartype
import jax.random
import jaxtyping as jt
import jraph

from . import _spatial

RandomFn = Callable[[jax.typing.ArrayLike, int], jax.typing.ArrayLike]
LiteralOrRandom = Union[jax.typing.ArrayLike, RandomFn]


@jt.jaxtyped(typechecker=beartype.beartype)
def spatial_graph(
    rng_key: jax.Array,
    num_nodes: int = None,
    num_graphs=None,
    cutoff=0.4,
    nodes: Optional[dict[str, LiteralOrRandom]] = None,
) -> Union[jraph.GraphsTuple, list[jraph.GraphsTuple]]:
    """Create graph(s) with nodes that have random positions"""
    graphs = []
    for _ in range(num_graphs or 1):
        if num_nodes is None:
            rng_key, subkey = jax.random.split(rng_key)
            num_nodes = jax.random.randint(subkey, shape=(), minval=2, maxval=10)

        rng_key, subkey = jax.random.split(rng_key)
        pos = jax.random.uniform(subkey, shape=(num_nodes, 3))

        if nodes is not None:
            for key, value in nodes.items():
                value, rng_key = _create_attributes(value, rng_key, num_nodes)
                nodes[key] = value

        graphs.append(_spatial.graph_from_points(pos, r_max=cutoff, nodes=nodes))

    if num_graphs is None:
        return graphs[0]

    return graphs


@jt.jaxtyped(typechecker=beartype.beartype)
def _create_attributes(
    value: LiteralOrRandom, rng_key: jax.Array, num: int
) -> tuple[jax.typing.ArrayLike, jax.Array]:
    if isinstance(value, Callable):
        rng_key, subkey = jax.random.split(rng_key)
        return value(subkey, num), rng_key

    return value, rng_key
