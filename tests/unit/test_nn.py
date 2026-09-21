"""Tests for `tensorial.nn.Sequential` and the private `_layers` helper.

Covers the public API (`__post_init__` validation, `__call__` behaviour for
each output type) and the internal `_layers` construction logic (partial
resolution, FrozenDict references, nested ``Sequential`` building).
"""

import functools

from flax import linen
import jax
import jax.numpy as jnp
import jraph
import pytest

import tensorial.nn
from tensorial.nn import Sequential, _layers

# ── helper modules ─────────────────────────────────────────────────────────────


class Identity(linen.Module):
    """Passes its (single) argument through unchanged."""

    def __call__(self, x):
        return x


class AddOne(linen.Module):
    @linen.compact
    def __call__(self, x):
        return x + 1.0


class TwoOutputs(linen.Module):
    """Returns a literal ``tuple`` of two arrays."""

    def __call__(self, x):
        return (x + 1.0, x * 3.0)


class SubclassDict(linen.Module):
    """Returns a ``dict`` so the next layer is called with ``**kwargs``."""

    def __call__(self, x):
        return {"a": x + 1.0, "b": x * 2.0}


class SumOfArgs(linen.Module):
    """Accepts keyword args ``a`` and ``b``, returns their sum."""

    def __call__(self, a, b):
        return a + b


class TakeTwoPositional(linen.Module):
    """Accepts two positional args, returns their sum."""

    def __call__(self, a, b):
        return a + b


class MyTuple(tuple):
    """A simple ``tuple`` subclass used to verify that tuple subclasses stay intact."""


class TupleSubclassOutput(linen.Module):
    """Returns a ``MyTuple`` instance (i.e. a tuple subclass, not a bare tuple)."""

    def __call__(self, x):
        return MyTuple((x + 1.0, x * 3.0))


class GraphsProducer(linen.Module):
    def __call__(self, x):
        return jraph.GraphsTuple(
            nodes=jnp.array([1.0, 2.0]),
            edges=jnp.array([3.0]),
            senders=jnp.array([0]),
            receivers=jnp.array([1]),
            globals=None,
            n_node=jnp.array([2]),
            n_edge=jnp.array([1]),
        )


class GraphsConsumer(linen.Module):
    """Expects a full ``GraphsTuple`` as its single argument — not unpacked."""

    @linen.compact
    def __call__(self, graph):
        return graph.nodes + 10.0


# ── __post_init__ validation ───────────────────────────────────────────────────


class TestPostInit:
    def test_non_sequence_raises(self):
        with pytest.raises(ValueError, match="must be a sequence"):
            Sequential(123)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="must be a sequence"):
            Sequential(None)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="Empty Sequential module"):
            Sequential([])

    def test_valid_single_layer(self):
        seq = Sequential([Identity()])
        assert seq.layers == [Identity()]

    def test_valid_multiple_layers(self):
        seq = Sequential([AddOne(), AddOne()])
        assert len(seq.layers) == 2


# ── __call__: single argument ──────────────────────────────────────────────────


class TestCallSingleArg:
    def test_single_layer_passthrough(self, rng_key):
        seq = Sequential([Identity()])
        params = seq.init(rng_key, jnp.array(1.5))
        out = seq.apply(params, jnp.array(1.5))
        assert out == 1.5

    def test_two_layers_chain(self, rng_key):
        seq = Sequential([AddOne(), AddOne()])
        params = seq.init(rng_key, jnp.array(1.0))
        out = seq.apply(params, jnp.array(1.0))
        assert out == 3.0

    def test_dense_chain_numerical(self, rng_key):
        """Chain of two Dense layers produces a numeric result of the right shape."""
        x = jnp.ones(3)
        seq = Sequential([linen.Dense(4), linen.Dense(2)])
        params = seq.init(rng_key, x)
        out = seq.apply(params, x)
        assert out.shape == (2,)
        assert jnp.isfinite(out).all()


# ── __call__: dict output → **kwargs ──────────────────────────────────────────


class TestCallDictOutput:
    def test_dict_forwarded_as_kwargs(self, rng_key):
        seq = Sequential([SubclassDict(), SumOfArgs()])
        params = seq.init(rng_key, jnp.array(5.0))
        out = seq.apply(params, jnp.array(5.0))
        # SubclassDict(5) → {a: 6, b: 10} → SumOfArgs(a=6, b=10) = 16
        assert out == 16.0


# ── __call__: plain tuple output → *args ──────────────────────────────────────


class TestCallPlainTupleOutput:
    def test_plain_tuple_fans_out_positionally(self, rng_key):
        """A bare ``tuple`` return from a layer is unpacked positionally (*args),
        matching flax's behaviour (and the module docstring)."""
        seq = Sequential([TwoOutputs(), TakeTwoPositional()])
        x = jnp.array(2.0)
        params = seq.init(rng_key, x)
        out = seq.apply(params, x)
        # TwoOutputs(2) = (3.0, 6.0) → TakeTwoPositional(3, 6) = 9
        assert out == 9.0


# ── __call__: tuple subclass stays intact ─────────────────────────────────────


class TestCallTupleSubclass:
    def test_tuple_subclass_stays_intact(self, rng_key):
        """A tuple subclass is NOT unpacked; the next layer receives it as a
        single positional argument."""
        seq = Sequential([TupleSubclassOutput(), MyTupleConsumer()])
        x = jnp.array(2.0)
        params = seq.init(rng_key, x)
        out = seq.apply(params, x)
        # TupleSubclassOutput(2) = MyTuple((3.0, 6.0)), passed whole
        # → MyTupleConsumer(t) = t[0] + t[1] = 9
        assert out == 9.0


class MyTupleConsumer(linen.Module):
    """Receives ``MyTuple`` as a single argument and sums its elements."""

    def __call__(self, t):
        return t[0] + t[1]


# ── __call__: GraphsTuple stays intact ─────────────────────────────────────────


class TestCallGraphsTuple:
    def test_graphs_tuple_kept_whole(self, rng_key):
        """``jraph.GraphsTuple`` is a tuple subclass; it must be forwarded as a
        single object, not unpacked into the next layer."""
        seq = Sequential([GraphsProducer(), GraphsConsumer()])
        params = seq.init(rng_key, jnp.array(0.0))
        out = seq.apply(params, jnp.array(0.0))
        # GraphsProducer → GraphsTuple(nodes=[1,2],…) → GraphsConsumer(graph)
        # → graph.nodes + 10 = [11, 12]
        assert jnp.allclose(out, jnp.array([11.0, 12.0]))


# ── _layers: module lists ──────────────────────────────────────────────────────


class TestLayers:
    def test_single_module_unwrapped(self):
        mod = Identity()
        result = _layers([mod])
        assert result == [mod]

    def test_two_modules_kept_as_list(self):
        a, b = AddOne(), AddOne()
        result = _layers([a, b])
        assert result == [a, b]

    def test_empty_list(self):
        assert _layers([]) == []


# ── _layers: partial + FrozenDict references ──────────────────────────────────


def _build_from_ref(ref):
    """Returns a module that stores ``ref`` as ``.value``."""

    class Holder(linen.Module):
        def __init__(self, ref):
            super().__init__()
            self.value = ref

    return Holder(ref)


def _partial_builder(ref):
    return _build_from_ref(ref)


class TestLayersPartialReferences:
    def test_partial_resolved_via_frozendict_reference(self):
        """A ``functools.partial`` whose unfilled param name matches a key in a
        preceding ``FrozenDict`` is called with that value."""
        ref = linen.FrozenDict({"ref": jnp.array([1.0])})
        result = _layers([ref, functools.partial(_partial_builder)])
        assert len(result) == 1
        assert jnp.allclose(result[0].value, jnp.array([1.0]))

    def test_partial_with_multiple_references_order(self):
        """Unfilled params are passed in *name order*, matching the reference dict."""

        class PairHolder(linen.Module):
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b

        def pair_builder(a, b):
            return PairHolder(a, b)

        # Key order in FrozenDict is irrelevant; param order is (a, b)
        ref = linen.FrozenDict({"b": jnp.array(99.0), "a": jnp.array(1.0)})
        result = _layers([ref, functools.partial(pair_builder)])
        assert result[0].a == 1.0
        assert result[0].b == 99.0

    def test_partial_reference_not_resolved_wraps_in_sequential(self):
        """When a partial's unfilled param has no matching FrozenDict reference but
        there are preceding modules, they are collected into a ``Sequential`` and
        passed to the partial."""

        class SeqHolder(linen.Module):
            def __init__(self, seq):
                super().__init__()
                self.seq = seq

        def seq_builder(seq):
            return SeqHolder(seq)

        m1, m2 = AddOne(), AddOne()
        result = _layers([m1, m2, functools.partial(seq_builder)])
        assert len(result) == 1
        assert isinstance(result[0], SeqHolder)
        # With >1 preceding module the holder stores a Sequential
        assert isinstance(result[0].seq, Sequential)

    def test_partial_single_preceding_module_passed_directly(self):
        """With exactly one preceding module it is passed directly, not wrapped."""

        class SeqHolder(linen.Module):
            def __init__(self, seq):
                super().__init__()
                self.seq = seq

        def seq_builder(seq):
            return SeqHolder(seq)

        m1 = AddOne()
        result = _layers([m1, functools.partial(seq_builder)])
        assert len(result) == 1
        # Single preceding module is passed bare
        assert result[0].seq is m1

    def test_partial_as_first_module_raises(self):
        with pytest.raises(ValueError, match="no previous modules"):
            _layers([functools.partial(_partial_builder)])

    def test_partial_resolving_to_non_module_raises(self):
        # Unfilled param ``seq`` has no matching reference, so the preceding
        # module is passed in; the result is not a Module → ValueError.
        def not_a_module(seq):
            return "not-a-module"

        with pytest.raises(ValueError, match="did not resolve"):
            _layers([AddOne(), functools.partial(not_a_module)])
