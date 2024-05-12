# -*- coding: utf-8 -*-
import math

import numpy as np
import pytest

from tensorial import data


@pytest.mark.parametrize("dataset_size,batch_size", ((27, 7), (4, 2)))
def test_array_loader(dataset_size, batch_size):
    num_batches = math.ceil(dataset_size / batch_size)

    inputs = np.random.rand(dataset_size)
    outputs = np.random.rand(dataset_size)

    # No outputs
    batches = tuple(data.ArrayLoader(inputs, batch_size=batch_size))
    assert len(batches) == num_batches
    assert isinstance(batches[0][0], type(inputs))
    assert np.allclose(batches[0][0], inputs[:batch_size])  # Check the first batch

    batches = tuple(data.ArrayLoader(inputs, outputs, batch_size=batch_size))
    assert isinstance(batches[0][0], type(inputs))
    assert len(batches) == num_batches
    assert len(batches[-1][0]) == dataset_size - (num_batches - 1) * batch_size


def test_caching_loader():
    inputs = np.arange(10)
    loader = data.CachingLoader(data.ArrayLoader(inputs, shuffle=True, batch_size=6), reset_every=3)
    batches = tuple(loader)
    # Check that the result is the same for two more iterations
    for _ in range(2):
        for idx, batch in enumerate(tuple(loader)):
            assert np.all(batches[idx][0] == batch[0])

    # Nw check that it changed
    for _ in range(3):
        for idx, batch in enumerate(tuple(loader)):
            assert np.any(batches[idx][0] != batch[0])
