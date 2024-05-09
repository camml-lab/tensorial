# -*- coding: utf-8 -*-
from typing import Type

import clu.metrics
import flax.struct
import jax
import jax.numpy as jnp
import jraph
from pytray import tree

import tensorial

from . import utils

__all__ = ("graph_metric",)


def graph_metric(
    metric: Type[clu.metrics.Metric],
    *field: utils.TreePathLike,
    _per_node=False,
    **kwargs: utils.TreePathLike,
) -> Type[clu.metrics.Metric]:
    arg_paths = tuple(map(utils.path_from_str, field))
    kwarg_paths = {key: utils.path_from_str(value) for key, value in kwargs.items()}
    mask = kwarg_paths.pop("mask", None)

    @flax.struct.dataclass
    class FromGraph(metric):
        """Wrapper Metric class that collects output named `name`."""

        @classmethod
        def from_model_output(cls, **out_kwargs) -> clu.metrics.Metric:
            # Extract the paths we are interested in
            from_args = []
            from_kwrags = {}
            mask_value = None

            # Args
            for path in arg_paths:
                field, graph_path = path[0], path[1:]
                graph: jraph.GraphsTuple = out_kwargs[field]
                assert isinstance(
                    graph, jraph.GraphsTuple
                ), f"Expected a GraphsTuple for model output {field}, got {type(graph).__name__}"

                graph_dict = graph._asdict()
                values = tensorial.as_array(tree.get_by_path(graph_dict, graph_path))

                if mask is not None:
                    mask_value = tensorial.as_array(tree.get_by_path(graph_dict, mask))

                if _per_node:
                    values = jax.vmap(jnp.divide, (0, 0))(values, jnp.maximum(graph.n_node, 1))

                from_args.append(values)

            # Kwargs
            for key, path in kwarg_paths.items():
                field, graph_path = path[0], path[1:]
                graph: jraph.GraphsTuple = out_kwargs[field]
                assert isinstance(
                    graph, jraph.GraphsTuple
                ), f"Expected a GraphsTuple for model output {field}, got {type(graph).__name__}"
                graph_dict = graph._asdict()
                values = tensorial.as_array(tree.get_by_path(graph_dict, graph_path))

                if mask is not None:
                    mask_value = tensorial.as_array(tree.get_by_path(graph_dict, mask))

                if _per_node:
                    values = jax.vmap(jnp.divide, (0, 0))(values, jnp.maximum(graph.n_node, 1))

                from_kwrags[key] = values

            if mask_value is not None:
                from_kwrags["mask"] = mask_value

            return super().from_model_output(*from_args, **from_kwrags)

    return FromGraph
