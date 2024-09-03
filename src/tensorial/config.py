import functools
from typing import Any, Dict, Union

from flax import linen
from flax.training import orbax_utils
import flax.training.train_state
import hydra
import jax
import omegaconf
import orbax.checkpoint
import reax

import tensorial

from . import data, metrics, modules

__all__ = ("create_module", "load_module_state")

Config = Union[omegaconf.DictConfig, omegaconf.ListConfig]

MODULE_STATE = "state"
MODULE_CONFIG = "config"
TRAIN_STATE = "train_state"


def create_module(module_config: Config) -> linen.Module:
    """Create the model from the configuration object"""
    if isinstance(module_config, omegaconf.ListConfig):
        mods = []
        for entry in module_config:
            mod = create_module(entry)
            if isinstance(mod, functools.partial):
                # We've reached a module that is partly constructed.  This indicates that it's a
                # module that wraps a function i.e. f(g(x)), typically because it needs access to
                # g(x) (for example to calculate gradients). So, we build what we've found so far,
                # and pass it to the module
                mod = mod(modules.Sequential(mods))
                if not isinstance(mod, linen.Module):
                    raise ValueError(
                        f"Calling partial module {type(mod).__name__}() did not resolve to a "
                        f"linen.Module instance"
                    )
                mods = [mod]
            else:
                mods.append(mod)

        if len(mods) == 1:
            # Special case to avoid needlessly wrapping a single module
            return mods[0]

        return modules.Sequential(mods)

    return hydra.utils.instantiate(module_config, _convert_="object")


def create_train_checkpoint(train_state: flax.training.train_state.TrainState):
    return {
        TRAIN_STATE: train_state,
    }


def create_module_checkpoint(module_config: Config, module_state) -> Dict:
    return {
        MODULE_CONFIG: omegaconf.OmegaConf.to_container(module_config, resolve=True),
        MODULE_STATE: module_state,
    }


def save_module(path, module_config: Config, module_state):
    save_state = create_module_checkpoint(module_config, module_state)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(
        path,
        save_state,
        save_args=orbax_utils.save_args_from_target(save_state),
    )


def load_module_state(path) -> tuple[Config, Any]:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = checkpointer.restore(path)
    return omegaconf.OmegaConf.create(state[MODULE_CONFIG]), state[MODULE_STATE]


def calculate_stats(from_data: omegaconf.DictConfig, training_data: data.DataLoader):
    """
    TODO: Update the name of this
    Update configuration with statistics gathered from the data

    :param from_data: the configuration dictionary to update (this will be done in place, i.e.
        overwrite the current value sof the dictionary)
    :param training_data: the training dataset to gather statistics from
    """
    coll_dict = {label: reax.metrics.get_registry()[name] for name, label in from_data.items()}
    collection = reax.metrics.MetricCollection(coll_dict)
    results = tensorial.metrics.Evaluator(collection).evaluate(training_data)

    # Update the configuration with the values we calculated
    for name, label in from_data.items():
        value = results[label]
        from_data[name] = value.tolist() if isinstance(value, jax.Array) else value
