import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import functools
from os import path

from flax import linen
from flax import linen as nn
import jax
import jax.numpy as jnp
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
import numpy as np
import optax
import torch

# from torch import nn
from torch import utils
from torch.utils.data import DataLoader, random_split

if _TORCHVISION_AVAILABLE:
    from torchvision.transforms import ToTensor

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")

from tensorial import jaxning as jn
from tensorial.jaxning.listeners import progress


class Autoencoder(linen.Module):
    hidden_dim: int = 64

    def setup(self):
        super().__init__()
        self.encoder = linen.Sequential([nn.Dense(self.hidden_dim), linen.relu, linen.Dense(3)])
        self.decoder = linen.Sequential(
            [nn.Dense(self.hidden_dim), linen.relu, linen.Dense(28 * 28)]
        )

    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LitAutoEncoder(jn.Module):
    def __init__(self, hidden_dim: int = 64, learning_rate=10e-3):
        super().__init__()
        self.autoencoder = Autoencoder(hidden_dim=hidden_dim)
        self._learning_rate = learning_rate

    def configure_model(self, batch):
        if self.parameters() is None:
            x = self._prepare_batch(batch)
            params = self.autoencoder.init(self.rng_key(), x)
            self.set_parameters(params)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        return self.autoencoder.apply(self.parameters(), x)

    def training_step(self, batch, batch_idx):
        x = self._prepare_batch(batch)
        loss, grads = jax.value_and_grad(self.loss_fn, argnums=0)(
            self.parameters(), x, self.autoencoder
        )

        self.log(f"train_loss", loss, on_step=True, prog_bar=True)
        return loss, grads

    @staticmethod
    @functools.partial(jax.jit, static_argnums=2)
    def loss_fn(params, x, model):
        predictions = model.apply(params, x)
        return optax.losses.squared_error(predictions, x).mean()

    def configure_optimizers(self):
        opt = optax.adam(learning_rate=self._learning_rate)
        state = opt.init(self.parameters())
        return opt, state

    @staticmethod
    def _prepare_batch(batch):
        x, _ = batch
        return jnp.asarray(x.view(x.size(0), -1).numpy())


if __name__ == "__main__":
    rng_key = jax.random.key(0)

    # setup data
    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset)

    autoencoder = LitAutoEncoder()

    trainer = jn.Trainer()
    trainer.events.add_listener(progress.TqdmProgressBar())
    trainer.fit(module=autoencoder, train_dataloaders=train_loader, max_epochs=1)
