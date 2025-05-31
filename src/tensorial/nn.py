import functools
from typing import Sequence, Union

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

    layers: Sequence[Union[linen.Module, functools.partial]]

    def setup(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        self._layers: list[linen.Module] = _layers(self.layers)

    def __post_init__(self):
        if not isinstance(self.layers, Sequence):
            raise ValueError(f"'layers' must be a sequence, got '{type(self.layers).__name__}'.")
        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")
        super().__post_init__()

    @linen.compact
    def __call__(self, *args, **kwargs):
        outputs = self._layers[0](*args, **kwargs)
        for layer in self._layers[1:]:
            if isinstance(outputs, dict):
                outputs = layer(**outputs)
            elif type(outputs) is tuple:  # pylint: disable=unidiomatic-typecheck
                outputs = layer(**outputs)
            else:
                outputs = layer(outputs)
        return outputs


def _layers(layers: Sequence[Union[linen.Module, functools.partial]]) -> list[linen.Module]:
    """Create the model from the configuration object"""
    new_layers: list[linen.Module] = []
    layers = list(layers)
    for i, layer in enumerate(layers):
        if isinstance(layer, functools.partial):
            # We've reached a module that is partly constructed.  This indicates that it's a
            # module that wraps a function i.e. f(g(x)), typically because it needs access to
            # g(x) (for example to calculate gradients). So, we build what we've found so far,
            # and pass it to the module
            # Wrap the rest of the layers in this sequence:
            to_wrap = layers[i + 1 :]
            if not to_wrap:
                raise ValueError(
                    f"Got a partial module, but have no subsequent modules to pass to it: {layer}"
                )
            wrapped = _layers(to_wrap)
            if len(wrapped) == 1:
                # Avoid wrapping a single layer, no point and makes debugging a little harder
                wrapped = wrapped[0]
            else:
                wrapped = Sequential(to_wrap)

            # Instantiate our partial
            layer = layer(wrapped)
            if not isinstance(layer, linen.Module):
                raise ValueError(
                    f"Calling partial module {type(layer).__name__}() did not resolve to a "
                    f"linen.Module instance"
                )

            new_layers.append(layer)
            break  # The rest have already been deal with above

        new_layers.append(layer)

    return new_layers
