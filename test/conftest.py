from jax import config, random
import pytest

# Some numerical test rely on checking against numpy implementations and np will default to 64-bit,
# so do that too
config.update("jax_enable_x64", True)


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)
