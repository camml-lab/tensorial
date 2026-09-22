from typing import Final

from flax import linen
import jax
import jax.numpy as jnp
import jaxtyping as jt
import jraph

from . import keys
from .. import gcnn, nn_utils
from .. import typing as tt
from .. import utils
from ..gcnn import atomic
from .keys import predicted

__all__ = ("CalcStresses",)

CellArray = jt.Float[tt.ArrayType, "n_graphs 3 3"]
DisplacementsArray = jt.Float[tt.ArrayType, "n_graphs 3 3"]
CELL_DISPLACEMENT: Final[str] = "cell_displacement"


class CalcStresses(linen.Module):
    """Module for computing virials and stresses from an energy function.

    This module wraps an energy function and computes the corresponding virial and
    stress tensors for each graph in a batch, adding them to the graph's node features.

    Attributes:
        energy_fn (mlip.GraphFunction): A function that computes the total energy from a graph.
        out_virials (mlip.TreePathLike | None): The key under which to store the computed virials
            in the graph's node dictionary. Defaults to `keys.VIRIALS`.
        out_stresses (mlip.TreePathLike | None): The key under which to store the computed
            stresses in the graph's node dictionary. Defaults to `keys.STRESSES`.
        energy_key (mlip.TreePathLike): The key used to retrieve the energy from the graph globals.
    """

    energy_fn: gcnn.GraphFunction
    out_virials: gcnn.TreePathLike | None = predicted(keys.VIRIAL)
    out_stresses: gcnn.TreePathLike | None = predicted(keys.STRESS)
    energy_key = predicted(atomic.keys.ENERGY)  # In the globals the energy will be found

    def setup(self):
        """Initializes the internal virial function used to compute stress-related quantities."""
        # pylint: disable=attribute-defined-outside-init
        self._virial_fn = gcnn.diff(
            self.virial_fn,
            f"globals.{keys.predicted(keys.TOTAL_ENERGY)}:gk",
            wrt=["1:gab"],
            out=":gab",
            return_graph=True,
        )

    def __call__(  # pylint: disable=arguments-differ
        self, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:
        """Computes and adds virials and stresses to the graph nodes.

        The virial tensor is computed via automatic differentiation of the energy
        with respect to strain. Stresses are obtained by dividing the virials
        by the cell volume, following the convention that
        `virial = -stress × volume`.

        Args:
            graph (jraph.GraphsTuple): A batched graph representing a set of atomistic systems.

        Returns:
            jraph.GraphsTuple: The input graph with updated node fields containing
            virials and optionally stresses.
        """
        if keys.CELL not in graph.globals:
            # Skip: no unit cell so can't calculate stresses, just pass to the energy function
            return self.energy_fn(graph)

        cell: CellArray = graph.globals[keys.CELL]
        cell_displacement: DisplacementsArray = self._symmetric_displacements(cell)
        virial, graph = self._virial_fn(graph, cell_displacement)

        updates = gcnn.experimental.update_graph(graph)
        if self.out_virials is not None:
            # From NequIP:
            # see discussion in https://github.com/libAtoms/QUIP/issues/227 about sign convention
            # they say the standard convention is virial = -stress x volume
            # this means that we need to pick up another negative sign for the virial
            # to fit this equation with the stress computed above
            updates.set(f"globals.{self.out_virials}", virial)

        if self.out_stresses is not None:
            volume = jax.vmap(gcnn.calc.cell_volume)(cell)
            if keys.MASK in graph.globals:
                mask = nn_utils.prepare_mask(graph.globals[keys.MASK], volume)
                volume = jnp.where(mask, volume, 1.0)

            stress = jax.vmap(jnp.divide)(virial, volume)
            updates.set(f"globals.{self.out_stresses}", stress)

        return updates.get()

    def virial_fn(
        self, graph: jraph.GraphsTuple, displacement: DisplacementsArray
    ) -> jraph.GraphsTuple:
        pos = graph.nodes[keys.POSITIONS]

        # Apply the symmetrized displacements to each atomic position
        all_displacements = jnp.repeat(
            displacement, graph.n_node, axis=0, total_repeat_length=pos.shape[0]
        )
        pos = pos + jax.vmap(jnp.matmul)(pos, all_displacements)

        # Update the graph
        graph = gcnn.experimental.update_graph(graph).set(f"nodes.{keys.POSITIONS}", pos).get()

        # Get the total energies
        return self.energy_fn(graph)

    @staticmethod
    def _symmetric_displacements(cells: CellArray, np_=None) -> CellArray:
        """Returns a symmetrized displacement tensor for each input cell.

        This function creates zero displacement tensors with the same shape as the input
        `cells` array and returns their symmetric part. Although the displacement is
        initialized to zero, the symmetry operation is still applied, which is useful for
        maintaining consistent computational structure in more complex code.

        Args:
            cells (jt.Float[typing.ArrayType, "n_graphs 3 3"]): A batch of 3x3 cell tensors,
                one for each graph in the batch.
            np_ (optional): A NumPy-compatible backend module (e.g., `numpy`, `jax.numpy`,
                or `torch`). If None, the backend is inferred from `cells`.

        Returns:
            jt.Float[typing.ArrayType, "n_graphs 3 3"]: The symmetric part of the displacement
            tensor, which is identically zero in this implementation.
        """
        if np_ is None:
            np_ = utils.infer_backend(cells)

        displacement = np_.zeros_like(cells)
        transpose = jax.vmap(jnp.transpose)
        return 0.5 * (displacement + transpose(displacement))
