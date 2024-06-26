import collections
import logging

import clu.metrics
from flax import linen
import jax
import optax

import tensorial
from tensorial import training


def create_test_validate(
    rng_key,
    n_train: int,
    n_validate: int,
    samples_per_batch=10,
    features=4,
) -> tuple[tensorial.data.ArrayLoader, tensorial.data.ArrayLoader]:
    keys = jax.random.split(rng_key, 5)

    train = tensorial.data.ArrayLoader(
        jax.random.uniform(keys[0], (n_train, samples_per_batch, features)),
        jax.random.uniform(keys[1], (n_train, samples_per_batch, features)),
    )
    validate = tensorial.data.ArrayLoader(
        jax.random.uniform(keys[2], (n_validate, samples_per_batch, features)),
        jax.random.uniform(keys[3], (n_validate, samples_per_batch, features)),
    )
    return train, validate


def test_trainer(rng_key):
    rng_key, *keys = jax.random.split(rng_key, 6)

    # Make up some data
    samples_per_batch = 10
    features = 4
    # [n batches, test/train, samples per batch, n features]
    train, validate = create_test_validate(rng_key, 3, 2, samples_per_batch, features)

    model = linen.linear.Dense(features=features)
    params = model.init(keys[4], train.first()[0])

    metrics = clu.metrics.Collection.create(
        loss=clu.metrics.Average.from_output("loss"),
        loss_std=clu.metrics.Std.from_output("loss"),
    )

    trainer = training.Trainer(
        model.apply,
        params,
        train_data=train,
        validate_data=validate,
        opt=optax.adam(learning_rate=1e-4),
        loss_fn=lambda x_pred, x_label: optax.losses.squared_error(x_pred, x_label).mean(),
        metrics=metrics,
    )

    num_epochs = 10
    assert trainer.train(max_epochs=num_epochs) == training.TRAIN_MAX_EPOCHS
    assert trainer.epoch == num_epochs
    # Check that metrics have been computed for teh last step
    assert "loss" in trainer.train_metrics
    assert "loss_std" in trainer.train_metrics

    assert "loss" in trainer.validate_metrics
    assert "loss_std" in trainer.validate_metrics


def test_trainer_early_stop(rng_key):
    # Make up some data
    samples_per_batch = 10
    features = 4
    # [n batches, test/train, samples per batch, n features]
    train, validate = create_test_validate(rng_key, 3, 2, samples_per_batch, features)

    # Dummy model that doesn't learn
    def model(_params, x):  # pylint: disable=invalid-name
        return x**2

    params = tuple()

    trainer = training.Trainer(
        model,
        params,
        train_data=train,
        validate_data=validate,
        opt=optax.adam(learning_rate=1e-4),
        loss_fn=lambda x_pred, x_label: optax.losses.squared_error(x_pred, x_label).mean(),
        overfitting_window=1,
    )
    assert trainer.train(max_epochs=3) == training.TRAIN_OVERFITTING


def test_trainer_logging(rng_key):
    rng_key, *keys = jax.random.split(rng_key, 4)

    # Make up some data
    samples_per_batch = 10
    features = 4
    # [n batches, test/train, samples per batch, n features]
    train, validate = create_test_validate(rng_key, 3, 2, samples_per_batch, features)

    model = linen.linear.Dense(features=features)
    params = model.init(keys[2], train.first()[0])

    trainer = training.Trainer(
        model.apply,
        params,
        train_data=train,
        validate_data=validate,
        opt=optax.adam(learning_rate=1e-4),
        loss_fn=lambda x_pred, x_label: optax.losses.squared_error(x_pred, x_label).mean(),
    )
    assert trainer.train(max_epochs=1) == training.TRAIN_MAX_EPOCHS
    log = trainer.metrics_log.raw_log()
    assert len(log) == 1

    assert log[0]["epoch"] == 0
    assert "training_loss" in log[0]
    assert "validation_loss" in log[0]


def test_metrics_logger():
    # Spoof a trainer
    Trainer = collections.namedtuple(
        "Trainer", ["metrics_log", "train_metrics", "validate_metrics"]
    )
    logger = training.TrainingLogger()
    train_metrics = {}
    validate_metrics = {}
    trainer = Trainer(logger, train_metrics, validate_metrics)

    lging = training.MetricsLogging(log_level=logging.WARNING, log_every=1)
    for epoch in range(2):
        train_metrics["loss"] = epoch**2
        validate_metrics["loss"] = epoch**3
        logger._save_log(trainer, epoch)  # pylint: disable=protected-access
        lging.on_epoch_finished(trainer, epoch)
