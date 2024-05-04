# -*- coding: utf-8 -*-
import math

import numpy as np
import pytest

from tensorial import datasets


@pytest.mark.parametrize("dataset_size,batch_size", ((27, 7), (4, 2)))
def test_generate_batches(dataset_size, batch_size):
    num_batches = math.ceil(dataset_size / batch_size)

    inputs = np.random.rand(dataset_size)
    outputs = np.random.rand(dataset_size)

    dataset = tuple(datasets.generate_batches(batch_size, inputs, outputs))
    assert len(dataset) == num_batches
    assert len(dataset[-1].inputs) == dataset_size - (num_batches - 1) * batch_size

    # No outputs
    dataset = tuple(datasets.generate_batches(batch_size, inputs))
    assert dataset[0].targets[0] is None


def test_generate_batches_batch_builder():
    dataset_size = 11
    batch_size = 4
    inputs = np.random.rand(dataset_size)
    outputs = np.random.rand(dataset_size)

    # If we only specify the input, then the output should default to the 'batch_builder' for inputs
    batches = tuple(datasets.generate_batches(batch_size, inputs, outputs, batch_builder=np.array))
    assert isinstance(batches[0].inputs, np.ndarray)
    assert isinstance(batches[0].targets, np.ndarray)

    # If we specify only the output batch builder, the input should remain the default
    batches = tuple(
        datasets.generate_batches(batch_size, inputs, outputs, output_batch_builder=np.array)
    )
    assert isinstance(batches[0].inputs, tuple)
    assert isinstance(batches[0].targets, np.ndarray)

    # Specify both separately
    batches = tuple(
        datasets.generate_batches(
            batch_size,
            inputs,
            outputs,
            batch_builder=list,
            output_batch_builder=np.array,
        )
    )
    assert isinstance(batches[0].inputs, list)
    assert isinstance(batches[0].targets, np.ndarray)
