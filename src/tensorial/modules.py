from typing import Any, Callable, Dict, Sequence

from flax import linen


class Sequential(linen.Module):
    """
    Applies a sequential chain of modules just like :class:`flax.linen.Sequential` _except_ that
    flax's version will expand any tuples that it receives when calling the next layer.  This
    doesn't play nice with types that subclass `tuple`, for example, :class:`jraph.GraphsTuple`,
    because the layers expect to get a `GraphsTuple`, not the individual values that make it up.

    Our behaviour is the same as :class:`flax.linen.Sequential` if we get a `tuple`, but any
    subclasses thereof are kept intact when calling the next layer.
    """

    layers: Sequence[Callable[..., Any]]

    def __post_init__(self):
        if not isinstance(self.layers, Sequence):
            raise ValueError(f"'layers' must be a sequence, got '{type(self.layers).__name__}'.")
        super().__post_init__()

    @linen.compact
    def __call__(self, *args, **kwargs):
        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")

        outputs = self.layers[0](*args, **kwargs)
        for layer in self.layers[1:]:
            if isinstance(outputs, Dict):
                outputs = layer(**outputs)
            elif type(outputs) is tuple:  # pylint: disable=unidiomatic-typecheck
                outputs = layer(**outputs)
            else:
                outputs = layer(outputs)
        return outputs
