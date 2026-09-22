import reax


def test_metrics_plugin():
    """Check that our plugins get automatically picked up by REAX"""
    registry = reax.metrics.get_registry()

    expected = [
        "atomic/energy_per_atom_rmse",
        "atomic/energy_per_atom_mae",
        "atomic/force_rmse",
        "atomic/stress_rmse",
    ]

    for metric_name in expected:
        assert metric_name in registry
