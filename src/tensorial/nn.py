"""Neural network building blocks.

Currently :class:`Sequential` only — a drop-in replacement for the
equivalent in :mod:`flax.linen` that preserves custom ``tuple`` subclasses
(e.g. :class:`jraph.GraphsTuple`) when passing values between layers.
"""

from collections.abc import Sequence
import functools
import inspect

from flax import linen


class Sequential(linen.Module):
    """Applies a sequential chain of modules just like :class:`flax.linen.Sequential` _except_ that
    flax's version will expand any tuples that it receives when calling the next layer.  This
    doesn't play nice with types that subclass `tuple`, for example, :class:`jraph.GraphsTuple`,
    because the layers expect to get a `GraphsTuple`, not the individual values that make it up.

    Our behaviour is the same as :class:`flax.linen.Sequential` if we get a `tuple`, but any
    subclasses thereof are kept intact when calling the next layer.
    """

    layers: Sequence[linen.Module | functools.partial]

    def setup(self) -> None:
        """Run the Flax :meth:`setup` hook, building the internal module list."""
        # pylint: disable=attribute-defined-outside-init
        self._layers: list[linen.Module] = _layers(self.layers)

    def __post_init__(self):
        """Validate ``layers`` on construction.

        Raises:
            ValueError: if ``layers`` is not a sequence, or is empty.
        """
        if not isinstance(self.layers, Sequence):
            raise ValueError(f"'layers' must be a sequence, got '{type(self.layers).__name__}'.")
        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")
        super().__post_init__()

    @linen.compact
    def __call__(self, *args, **kwargs):
        """Run each layer in turn, forwarding the previous layer's output.

        Args:
            *args: positional inputs forwarded to the first layer.
            **kwargs: keyword inputs forwarded to the first layer.

        Returns:
            the final layer's output.
        """
        outputs = self._layers[0](*args, **kwargs)
        for layer in self._layers[1:]:
            if isinstance(outputs, dict):
                outputs = layer(**outputs)
            elif type(outputs) is tuple:  # pylint: disable=unidiomatic-typecheck
                outputs = layer(**outputs)
            else:
                outputs = layer(outputs)
        return outputs


def _layers(modules: Sequence[linen.Module | functools.partial]) -> list[linen.Module]:
    """Create the model from the configuration object by building modules sequentially.

    Handles three cases:
    1. Regular modules: appended directly
    2. ``functools.partial`` modules: partially applied with reference arguments from
       preceding modules (when required parameters are available)
    3. ``linen.FrozenDict`` references: accumulated parameter references for later use

    The function supports constructing nested sequential modules when multiple partial
    modules depend on different earlier modules.

    Args:
        modules: A sequence of linen.Module instances or functools.partial objects.
                 If a partial module's required parameters are available from earlier
                 modules in the sequence, those are passed as arguments.

    Returns:
        A list of constructed linen.Module instances.
    """
    new_modules = []
    references = {}

    for module in modules:
        if isinstance(module, functools.partial):
            # We've reached a module that is partly constructed.  This indicates that it's a
            # module that wraps a function i.e. f(g(x)), typically because it needs access to
            # g(x) (for example to calculate gradients). So, we build what we've found so far,
            # and pass it to the module
            sig = inspect.signature(module)
            unfilled = [
                name
                for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty
            ]
            have_references = all(name in references for name in unfilled)
            if have_references:
                args = [references[name] for name in unfilled]
                new_modules.append(module(*args))
            else:
                if len(new_modules) == 0:
                    raise ValueError(
                        "Got a partial module, but have no previous modules to pass to it: "
                        f"{module}"
                    )

                if len(new_modules) == 1:
                    nested = new_modules[0]
                else:
                    nested = Sequential(new_modules)

                module = module(nested)
                if not isinstance(module, linen.Module):
                    raise ValueError(
                        f"Calling partial module {type(module).__name__}() did not resolve to a "
                        f"linen.Module instance"
                    )

                new_modules = [module]
        elif isinstance(module, linen.FrozenDict):
            references.update(module)
        else:
            new_modules.append(module)

    if len(new_modules) == 1:
        # Special case to avoid needlessly wrapping a single module
        return [new_modules[0]]

    return new_modules
