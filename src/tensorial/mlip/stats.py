from collections.abc import Callable, Sequence

import equinox as eqx
import jraph
from pytray import tree
import reax
from typing_extensions import override

_BoolReduce = Callable[[Sequence], bool]
_Oper = str | _BoolReduce


class Unary(reax.Metric[bool]):
    """A unary operator metric"""

    oper: _BoolReduce = eqx.field(static=True)
    _state: bool | None

    def __init__(self, oper: _Oper, state: bool = True):
        if oper == "and":
            self.oper = all
        elif oper == "or":
            self.oper = any
        elif isinstance(oper, Callable):
            self.oper = oper
        else:
            raise ValueError("operator must be 'and' or 'or' or a Callable")

        self._state: bool = state

    @override
    def create(self, values: Sequence, /, *_, **__) -> "Unary":
        return type(self)(self.oper, self.oper(values))

    @override
    def update(self, values: Sequence, /, *_, **__) -> "Unary":
        return self.create([self._state, *values])

    @override
    def merge(self, other: "Unary") -> "Unary":
        return type(self)(
            self.oper, self.oper([self._state, other._state])  # pylint: disable=protected-access
        )

    @override
    def compute(self) -> bool:
        return self._state


class Has(reax.Metric[bool]):
    _field: tuple = eqx.field(static=True)
    _state: Unary | None

    def __init__(self, field: str | tuple, oper: _Oper = "and", state: Unary | None = None):
        self._field = field if isinstance(field, tuple) else tuple(field.split("."))
        self._state = Unary(oper) if state is None else state

    @override
    def create(self, graphs: jraph.GraphsTuple, /, *_, **__) -> "Has":
        graph_dict = graphs._asdict()
        try:
            tree.get_by_path(graph_dict, self._field)
            state = self._state.create([True])
        except KeyError:
            state = self._state.create([False])

        return Has(self._field, state.oper, state)

    @override
    def update(self, graphs: jraph.GraphsTuple, /, *_, **__) -> "Has":
        graph_dict = graphs._asdict()
        try:
            tree.get_by_path(graph_dict, self._field)
            state = self._state.create([True])
        except KeyError:
            state = self._state.create([False])

        return Has(self._field, self._state.oper, self._state.merge(state))

    @override
    def merge(self, other: "Has") -> "Has":
        return type(self)(
            self._field,
            self._state.oper,
            self._state.merge(other._state),  # pylint: disable=protected-access
        )

    @override
    def compute(self) -> bool:
        return self._state.compute()
