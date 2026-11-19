import collections

import hydra
import omegaconf
import pytest

import tensorial
from tensorial import config

requires_whitelist = pytest.mark.skipif(
    not config.HAS_EXECUTION_WHITELIST, reason="Hydra < 1.4 has no execution whitelist"
)


@pytest.fixture(autouse=True)
def _clean_whitelist(monkeypatch):
    monkeypatch.delenv(config.WHITELIST_ENV_VAR, raising=False)
    monkeypatch.setattr(config, "_extra_whitelist", [])


def test_instantiate_whitelisted():
    cfg = omegaconf.OmegaConf.create({"_target_": "tensorial.tensors.OneHot", "num_classes": 3})
    one_hot = config.instantiate(cfg)
    assert isinstance(one_hot, tensorial.tensors.OneHot)


@requires_whitelist
def test_instantiate_not_whitelisted_raises():
    cfg = omegaconf.OmegaConf.create({"_target_": "collections.OrderedDict"})
    with pytest.raises(hydra.errors.InstantiationException):
        config.instantiate(cfg)


@requires_whitelist
def test_allow():
    cfg = omegaconf.OmegaConf.create({"_target_": "collections.OrderedDict"})
    config.allow("collections.*")
    assert "collections.*" in config.get_whitelist()
    assert isinstance(config.instantiate(cfg), collections.OrderedDict)


@requires_whitelist
def test_env_var(monkeypatch):
    cfg = omegaconf.OmegaConf.create({"_target_": "collections.OrderedDict"})
    monkeypatch.setenv(config.WHITELIST_ENV_VAR, "collections.OrderedDict, foo.*")
    whitelist = config.get_whitelist()
    assert "collections.OrderedDict" in whitelist
    assert "foo.*" in whitelist
    assert isinstance(config.instantiate(cfg), collections.OrderedDict)


@requires_whitelist
def test_env_var_unsafe(monkeypatch):
    monkeypatch.setenv(config.WHITELIST_ENV_VAR, "UNSAFE")
    assert config.get_whitelist() is config.UNSAFE_DISABLE_EXECUTION_CHECKS
    cfg = omegaconf.OmegaConf.create({"_target_": "collections.OrderedDict"})
    assert isinstance(config.instantiate(cfg), collections.OrderedDict)


def test_hydra_main_kwargs():
    kwargs = config.hydra_main_kwargs()
    if config.HAS_EXECUTION_WHITELIST:
        assert kwargs == {"execution_whitelist": config.get_whitelist()}
    else:
        assert not kwargs
