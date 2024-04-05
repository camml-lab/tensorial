# -*- coding: utf-8 -*-
from typing import Any, Dict, Tuple, Union

from flax import linen
from flax.training import orbax_utils
import flax.training.train_state
import hydra
import omegaconf
import orbax.checkpoint

from . import modules

__all__ = ('create_module', 'load_module_state')

Config = Union[omegaconf.DictConfig, omegaconf.ListConfig]

MODULE_STATE = 'state'
MODULE_CONFIG = 'config'
TRAIN_STATE = 'train_state'


def create_module(module_config: Config) -> linen.Module:
    """Create the model from the configuration object"""
    if isinstance(module_config, omegaconf.ListConfig):
        mods = []
        for entry in module_config:
            mods.append(create_module(entry))

        return modules.Sequential(mods)

    return hydra.utils.instantiate(module_config, _convert_='object')


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


def load_module_state(path) -> Tuple[Config, Any]:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = checkpointer.restore(path)
    return omegaconf.OmegaConf.create(state[MODULE_CONFIG]), state[MODULE_STATE]
