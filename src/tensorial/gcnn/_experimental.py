import copy
from typing import Any

import jraph


class GraphMutator:
    def __init__(self, graph: jraph.GraphsTuple):
        self.original = graph
        self.mutations: list[tuple[str, tuple[str | int, ...], Any]] = []

    def _normalize_path(self, path: str | tuple) -> tuple:
        return tuple(path.split(".")) if isinstance(path, str) else path

    def set(self, path: str | tuple, value) -> "GraphMutator":
        self.mutations.append(("set", self._normalize_path(path), value))
        return self

    def update(self, path: str | tuple, updates: dict) -> "GraphMutator":
        self.mutations.append(("update", self._normalize_path(path), updates))
        return self

    def delete(self, path: str | tuple) -> "GraphMutator":
        self.mutations.append(("delete", self._normalize_path(path), None))
        return self

    def _apply_mutation(self, container: Any, path: tuple, op: str, value: Any):
        if not path:
            if op == "set":
                return value

            if op == "update":
                if not isinstance(container, dict):
                    raise TypeError("Can only apply 'update' to a dict at root level.")
                container.update(value)
                return container

            if op == "delete":
                raise ValueError("Cannot delete the root container.")

            raise ValueError(f"Unsupported op '{op}' at root.")

        *parents, last = path
        target = container if container is not None else {}
        for key in parents:
            target = target[key]

        if op == "set":
            target[last] = value
        elif op == "update":
            if not isinstance(target[last], dict):
                raise TypeError(f"Cannot update non-dict object at path: {'.'.join(path)}")
            target[last].update(value)
        elif op == "delete":
            del target[last]

        return target if container is None else None

    def get(self) -> jraph.GraphsTuple:
        # Deepcopy all mutable fields; assume non-dict fields are safe to mutate directly
        mutated_fields = {
            # In gcnn these can be (mutable) dicts of immutable values
            "nodes": copy.copy(self.original.nodes),
            "edges": copy.copy(self.original.edges),
            "globals": copy.copy(self.original.globals),
            # while the following are simply immutable arrays
            "n_node": self.original.n_node,
            "n_edge": self.original.n_edge,
            "senders": self.original.senders,
            "receivers": self.original.receivers,
        }

        for op, path, value in self.mutations:
            root = path[0]
            if root not in mutated_fields:
                raise ValueError(f"Invalid root field '{root}' in mutation path.")
            result = self._apply_mutation(mutated_fields[root], path[1:], op, value)
            if result is not None:
                mutated_fields[root] = result

        return jraph.GraphsTuple(**mutated_fields)


def update_graph(graph: jraph.GraphsTuple) -> GraphMutator:
    return GraphMutator(graph)
