import optax

from . import keys
from .. import gcnn
from .keys import predicted


def mlip_loss(
    energy: bool = True,
    forces: bool = True,
    energy_weight: float = 1.0,
    force_weight: float = 20.0,
) -> gcnn.GraphLoss:
    weights: list[float] = []
    loss_terms: list[gcnn.GraphLoss] = []

    if energy:
        weights.append(energy_weight)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error,
                f"globals.{keys.TOTAL_ENERGY}",
                f"globals.{predicted(keys.TOTAL_ENERGY)}",
            )
        )

    if forces:
        weights.append(force_weight)
        loss_terms.append(
            gcnn.Loss(
                optax.squared_error, f"nodes.{keys.FORCES}", f"nodes.{predicted(keys.FORCES)}"
            )
        )

    if not loss_terms:
        raise ValueError(
            "Could not create loss function because all terms (energy, forces, ...) are set to "
            "`False`"
        )

    if len(loss_terms) == 1:
        return loss_terms[0]

    return gcnn.WeightedLoss(loss_terms, weights)
