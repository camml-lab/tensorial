# -*- coding: utf-8 -*-
import numpy as np
import omegaconf

from tensorial import config, data, metrics


def test_calculate_stats():
    n_points = 100
    array = np.random.rand(n_points, 2)
    dataset = data.ArrayLoader(array, batch_size=32)

    # Get all the metrics that calculate statistics about a dataset (rather than data/label)
    # this need not be an exhaustive list as not all need be subclasses of ``SimpleMetric``, but
    # it's a good start
    stats = {}
    for name, metric in metrics.registry.items():
        if issubclass(metric, metrics.StatMetric):
            stats[name] = f"data_{name}"

    from_data = omegaconf.DictConfig(stats)

    res = config.calculate_stats(from_data, dataset)
    # Just check it doesn't return anything, should update in place
    assert res is None

    for name, value in from_data.items():
        res = metrics.metric(name).from_model_output(array).compute()
        assert np.allclose(res, np.array(value))
