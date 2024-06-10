# -*- coding: utf-8 -*-
from __future__ import annotations  # For py39

import numpy as np
import pytest

from tensorial.data import samplers


def test_sequential_sampler():
    indices = np.arange(10).tolist()
    sampler = samplers.SequentialSampler(len(indices))
    assert list(sampler) == indices

    with pytest.raises(TypeError):
        samplers.SequentialSampler(iter(indices))


def test_random_sampler():
    indices = np.arange(10).tolist()
    sampler = samplers.RandomSampler(len(indices))
    assert sorted(list(sampler)) == indices

    sampler = samplers.RandomSampler(len(indices), replacements=True)
    assert all(sample in indices for sample in list(sampler))
