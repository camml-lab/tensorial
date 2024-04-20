# -*- coding: utf-8 -*-
from typing import Dict, Tuple, Type

import clu.metrics
import flax.struct
import jraph
from pytray import tree

import tensorial

from . import utils

__all__ = ('graph_metric',)


def graph_metric(metric: Type[clu.metrics.Metric], *field: Tuple[str], **kwargs: Dict[str,
                                                                                      str]) -> Type[clu.metrics.Metric]:
    arg_paths = tuple(map(utils.path_from_str, field))
    kwarg_paths = {key: utils.path_from_str(value) for key, value in kwargs.items()}

    @flax.struct.dataclass
    class FromGraph(metric):
        """Wrapper Metric class that collects output named `name`."""

        @classmethod
        def from_model_output(cls, **out_kwargs) -> clu.metrics.Metric:
            # Extract the paths we are interested in
            from_args = []
            for path in arg_paths:
                field, graph_path = path[0], path[1:]
                graph: jraph.GraphsTuple = out_kwargs[field]
                assert isinstance(
                    graph, jraph.GraphsTuple
                ), f'Expected a GraphsTuple for model output {field}, got {type(graph).__name__}'
                from_args.append(tensorial.as_array(tree.get_by_path(graph._asdict(), graph_path)))

            from_kwrags = {}
            for key, path in kwarg_paths.items():
                field, graph_path = path[0], path[1:]
                graph: jraph.GraphsTuple = out_kwargs[field]
                assert isinstance(
                    graph, jraph.GraphsTuple
                ), f'Expected a GraphsTuple for model output {field}, got {type(graph).__name__}'
                from_kwrags[key] = tensorial.as_array(tree.get_by_path(graph._asdict(), graph_path))

            return super().from_model_output(*from_args, **from_kwrags)

    return FromGraph
