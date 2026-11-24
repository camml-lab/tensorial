from typing import cast

import pytest

from tensorial import gcnn
from tensorial.mlip import mlip_loss


def _terms(loss: gcnn.WeightedLoss) -> dict[str, bool]:
    return {loss_fn.label(): cast(gcnn.Loss, loss_fn).optional for loss_fn in loss._loss_fns}


def test_mlip_loss_terms_are_optional_by_default():
    loss = mlip_loss()
    assert _terms(loss) == {
        "globals.predicted_energy": True,
        "nodes.predicted_forces": True,
        "globals.predicted_stress": True,
    }
    assert loss._weights == (1.0, 20.0, 1.0)


def test_mlip_loss_required_and_excluded_terms():
    loss = mlip_loss(energy=True, forces=False, stress=None, stress_weight=0.5)
    assert _terms(loss) == {"globals.predicted_energy": False, "globals.predicted_stress": True}
    assert loss._weights == (1.0, 0.5)


def test_mlip_loss_no_terms():
    with pytest.raises(ValueError, match="all terms"):
        mlip_loss(energy=False, forces=False, stress=False)


@pytest.mark.parametrize("energy_per_atom,expected", [(True, ("n_node",)), (False, None)])
def test_mlip_loss_energy_per_atom(energy_per_atom, expected):
    loss = mlip_loss(energy_per_atom=energy_per_atom)
    energy_term = cast(gcnn.Loss, loss._loss_fns[0])
    assert energy_term._normalise_by == expected
