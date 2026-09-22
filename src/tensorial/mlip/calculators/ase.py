import functools
import pathlib
from typing import Final

import ase
from ase.calculators import calculator
import jax
import jaxtyping as jt
import jraph
import numpy as np
import omegaconf
from pytray import tree
import reax
from typing_extensions import override

from .. import experimental, keys, typing
from ... import config, gcnn
from ...gcnn import _tree, atomic


class Calculator(calculator.Calculator):
    """
    E3MD's ASE calculator.
    """

    def __init__(
        self,
        model: typing.Model,
        params: jt.PyTree,
        r_max: float,
        accelerator: str = "auto",
        jit=True,
        atom_include_keys=tuple(),
    ):
        super().__init__()
        self.implemented_properties = ["energy", "forces", "stress"]

        # Params
        self._model: Final[typing.Model] = model
        self._params: Final[jt.PyTree] = params
        self._r_max: Final[float] = r_max
        self._atom_include_keys = atom_include_keys + ("numbers",)

        if jit:
            calc = functools.partial(jax.jit(self._model.apply), self._params)
        else:
            calc = functools.partial(self._model.apply, self._params)
        self._calculate = calc
        self._device: jax.Device = (
            jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        )

        # State
        self._graph_padding: gcnn.data.GraphPadding | None = None

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | pathlib.Path,
        checkpoint_path: str | pathlib.Path,
        checkpointing: reax.training.Checkpointing = None,
        **kwargs,
    ) -> "Calculator":
        cfg = omegaconf.OmegaConf.load(config_path)
        model: typing.Model = config.instantiate(cfg["model"]["model"])
        if checkpointing is None:
            checkpointing = reax.training.get_default_checkpointing()

        ckpt = checkpointing.load(checkpoint_path)
        params = ckpt[reax.training._checkpointing.PARAMS]  # pylint: disable=protected-access

        return cls(model, params, r_max=cfg["r_max"], **kwargs)

    @property
    def graph_padding(self) -> gcnn.data.GraphPadding | None:
        return self._graph_padding

    @override
    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=("energy",),
        system_changes=tuple(calculator.all_changes),
    ):
        super().calculate(atoms)

        graph = self._create_graph(atoms)
        graph = jax.device_put(graph, self._device)
        out = self._calculate(graph)

        self.results = {}

        energy = out.globals[keys.predicted(keys.TOTAL_ENERGY)]
        stress = out.globals[keys.predicted(keys.STRESS)]

        if keys.MASK in out.globals:
            energy = energy[out.globals[keys.MASK]]
            stress = stress[out.globals[keys.MASK]]

        energy = energy.array.item()
        stress = np.array(stress)

        forces = out.nodes[keys.predicted(keys.FORCES)]
        if keys.MASK in out.nodes:
            forces = forces[out.nodes[keys.MASK]]
        forces = np.array(forces)

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress

    def preprocess(self, structures: list[ase.Atoms]):
        max_nodes = 0
        max_edges = 0

        for structure in structures:
            graph: jraph.GraphsTuple = atomic.graph_from_ase(
                structure, self._r_max, use_calculator=False
            )
            max_nodes = max(max_nodes, graph.n_node[0].item())
            max_edges = max(max_edges, graph.n_edge[0].item())

        self._graph_padding = gcnn.data.GraphPadding(max_nodes + 1, max_edges, 2)

    def _create_graph(self, atoms: ase.Atoms) -> jraph.GraphsTuple:
        graph = atomic.graph_from_ase(
            atoms,
            self._r_max,
            use_calculator=False,
            atom_include_keys=self._atom_include_keys,
        )
        if self._graph_padding is not None:
            graph = gcnn.data.pad_with_graphs(graph, *self._graph_padding)

        return graph


class AdiabaticCalculator(Calculator):
    def __init__(
        self,
        model: typing.Model,
        params: jt.PyTree,
        r_max: float,
        to_minimize: gcnn.TreePathLike,
        accelerator: str = "auto",
        jit=True,
        atom_include_keys=tuple(),
    ):
        super().__init__(
            model,
            params,
            r_max,
            accelerator=accelerator,
            jit=jit,
            atom_include_keys=atom_include_keys,
        )
        self._to_minimize = _tree.path_from_str(to_minimize)
        calc = functools.partial(jax.jit(self._model.apply), self._params)

        def minim_cal(graph):
            graph = calc(graph)
            total_energy = graph.globals[keys.predicted(keys.TOTAL_ENERGY)]
            globals_dict = graph.globals
            penalty = 1e6 * graph.nodes["spins"].sum() ** 2
            globals_dict[keys.predicted(keys.TOTAL_ENERGY)] = total_energy + penalty
            return graph._replace(globals=globals_dict)

        min_fn = experimental.minimize_fn(
            minim_cal, f"globals.{keys.predicted(keys.TOTAL_ENERGY)}", to_minimize
        )
        if jit:
            min_fn = jax.jit(min_fn, static_argnames=("method",))
        self._minimize_fn = min_fn
        self.implemented_properties = list(self.implemented_properties) + [self._to_minimize[-1]]

    @override
    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties=None,
        system_changes=tuple(calculator.all_changes),
    ):
        minimizing = self._to_minimize[-1]
        if properties is None:
            properties = ("energy", minimizing)

        graph = self._create_graph(atoms)
        graph = jax.device_put(graph, self._device)
        path = _tree.path_from_str(self._to_minimize)
        x0 = tree.get_by_path(graph._asdict(), path)

        res = self._minimize_fn(graph, x0, method="BFGS")
        optimized_spins = np.array(res.x)

        # Update the Atoms object with the value at the minimum
        atoms.set_initial_magnetic_moments(optimized_spins)

        # Now calculate the energies and forces
        super().calculate(atoms, properties, system_changes)

        # Also save the spins in the results
        self.results[minimizing] = optimized_spins
