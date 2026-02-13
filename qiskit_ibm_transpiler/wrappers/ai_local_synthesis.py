# -*- coding: utf-8 -*-

# (C) Copyright 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx
import numpy as np
from networkx.exception import NetworkXError
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.providers.backend import BackendV2 as Backend
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap

from ..model_repository import RLSynthesisRepository
from ..utils import embed_clifford

logger = logging.getLogger(__name__)

DEFAULT_NUM_SEARCHES = 10
logger.setLevel(logging.INFO)


def validate_coupling_map_source(coupling_map, backend):
    if not coupling_map and not backend:
        raise ValueError(
            "ERROR. Either a 'coupling_map' or a 'backend' must be provided."
        )


def validate_circuits_and_qargs_lengths(circuits: list, qargs: list):
    n_circs = len(circuits)
    n_qargs = len(qargs)

    if n_circs != n_qargs:
        raise ValueError(
            f"The number of input circuits {n_circs}"
            f"and the number of input qargs arrays {n_qargs} are different."
        )


def get_formatted_coupling_map(coupling_map):
    formatted_coupling_map = None
    if coupling_map:
        if isinstance(coupling_map, CouplingMap):
            formatted_coupling_map = coupling_map
        elif isinstance(coupling_map, list):
            formatted_coupling_map = CouplingMap(couplinglist=coupling_map)
        else:
            raise ValueError(
                f"ERROR. coupling_map should either be a list of int tuples or a Qiskit CouplingMap object."
            )

    return formatted_coupling_map


def get_coupling_map_graph(
    backend: Backend = None, coupling_map: CouplingMap = None
) -> nx.Graph:
    backend_coupling_map = (
        coupling_map if getattr(coupling_map, "graph", None) else backend.coupling_map
    )

    coupling_map_edges = backend_coupling_map.get_edges()
    coupling_map_list_format = [list(edge) for edge in coupling_map_edges]

    try:
        coupling_map = nx.Graph(coupling_map_list_format)
    except Exception:
        raise NetworkXError("ERROR. Cannot convert coupling_map from list to graph")

    return coupling_map


def get_mapping_perm(
    coupling_map: nx.Graph, circuit_qargs: list[int], coupling_maps_by_hashes: dict
) -> list[int]:
    # Identify the subgraph of the device coupling map where the circuit is.
    circuit_in_coupling_map = coupling_map.subgraph(circuit_qargs)

    is_circuit_in_coupling_map_connected = nx.is_connected(circuit_in_coupling_map)

    if not is_circuit_in_coupling_map_connected:
        raise ValueError(
            "ERROR. Qargs do not form a connected subgraph of the backend coupling map"
        )

    # We find which model to use by hashing the circuit subgraph
    circuit_in_coupling_map_hash = nx.weisfeiler_lehman_graph_hash(
        circuit_in_coupling_map
    )

    # If there is no model for that circuit_in_coupling_map, we cannot use AI.
    if circuit_in_coupling_map_hash not in coupling_maps_by_hashes:
        raise LookupError("ERROR. No model available for the requested subgraph")

    if isinstance(coupling_maps_by_hashes, RLSynthesisRepository):
        model_coupling_map = coupling_maps_by_hashes.get(
            circuit_in_coupling_map_hash
        ).coupling_map
    else:
        model_coupling_map = coupling_maps_by_hashes[circuit_in_coupling_map_hash]

    # circuit_in_coupling_map could appears several times on the model's topology. We get the first
    # we find due to the `next` function. device_model_mapping contains a nodes correspondency between
    # the device and the model
    device_model_mapping = next(
        iter(
            nx.isomorphism.GraphMatcher(
                circuit_in_coupling_map, nx.Graph(model_coupling_map)
            ).match()
        )
    )

    # We now have to find the permutation that we should apply to the Clifford based on the mapping we found
    qargs_dict = {v: i for i, v in enumerate(circuit_qargs)}

    # subgraph_perm will be a list where each position indicates a permutation. The index refers to the circuit
    # qubit and the value refers to the qubit on the model
    subgraph_perm = [
        qargs_dict[v]
        for v in sorted(
            device_model_mapping.keys(), key=lambda k: device_model_mapping[k]
        )
    ]

    return subgraph_perm, circuit_in_coupling_map_hash


def perm_cliff(cliff, perm):
    perm = np.array(perm)
    cliff.stab_x = cliff.stab_x[:, perm]
    cliff.stab_z = cliff.stab_z[:, perm]
    cliff.destab_x = cliff.destab_x[:, perm]
    cliff.destab_z = cliff.destab_z[:, perm]
    cliff.stab = cliff.stab[perm, :]
    cliff.destab = cliff.destab[perm, :]
    return cliff


class AILocalSynthesisBase(ABC):
    """Abstract base class for local-mode AI synthesis backed by cached RL models."""

    synthesis_type: str = "base"

    def __init__(
        self,
        model_repo: RLSynthesisRepository,
        num_searches: int = DEFAULT_NUM_SEARCHES,
    ) -> None:
        self.model_repo = model_repo
        self.num_searches = num_searches

    @abstractmethod
    def _prepare_input(
        self,
        circuit: Any,
        subgraph_perm: list[int],
        target_qubits: int,
    ) -> Any | None:
        """Prepare input for synthesis with permutation and embedding.

        Args:
            circuit: The input circuit or object to prepare.
            subgraph_perm: The permutation to apply.
            target_qubits: The target number of qubits for the model.

        Returns:
            The prepared input or None if preparation fails.
        """
        pass

    def _get_input_n_qubits(self, circuit: Any) -> int:
        """Get the number of qubits from the input.

        Override this method for types that don't have num_qubits attribute.
        """
        return circuit.num_qubits

    def _synthesize_circuits(
        self,
        coupling_map: nx.Graph,
        circuits: list[Any],
        qargs: list[list[int]],
    ) -> list[QuantumCircuit | None]:
        """Synthesize circuits using qiskit-gym models.

        Args:
            coupling_map: The coupling map graph.
            circuits: List of circuits or objects to synthesize.
            qargs: List of qubit indices for each circuit.

        Returns:
            List of synthesized QuantumCircuits or None for failed syntheses.
        """
        synthesized_circuits: list[QuantumCircuit | None] = []

        for circuit, circuit_qargs in zip(circuits, qargs):
            try:
                subgraph_perm, cmap_hash = get_mapping_perm(
                    coupling_map, circuit_qargs, self.model_repo
                )
            except Exception as exc:
                logger.warning(exc)
                synthesized_circuits.append(None)
                continue

            try:
                record = self.model_repo.get(cmap_hash)
            except KeyError:
                logger.warning(
                    "%s model for hash %s is not registered",
                    self.synthesis_type,
                    cmap_hash,
                )
                synthesized_circuits.append(None)
                continue

            model = record.model
            model_n_qubits = int(model.env_config.get("num_qubits", len(circuit_qargs)))

            prepared_input = self._prepare_input(circuit, subgraph_perm, model_n_qubits)
            if prepared_input is None:
                synthesized_circuits.append(None)
                continue

            try:
                synthesized = model.synth(
                    input=prepared_input, num_searches=self.num_searches
                )
            except Exception as err:
                logger.warning(
                    "%s synthesis failed for hash %s: %s",
                    self.synthesis_type,
                    cmap_hash,
                    err,
                )
                synthesized_circuits.append(None)
                continue

            if synthesized is None:
                synthesized_circuits.append(None)
                continue

            output_circuit = QuantumCircuit(synthesized.num_qubits).compose(
                synthesized, qubits=subgraph_perm
            )
            synthesized_circuits.append(output_circuit)

        return synthesized_circuits

    def transpile(
        self,
        circuits: list[Any],
        qargs: list[list[int]],
        coupling_map: list[list[int]] | CouplingMap | None = None,
        backend_name: str | None = None,
        backend: Backend | None = None,
    ) -> list[QuantumCircuit | None]:
        """Synthesize one or more circuits into optimized equivalents.

        It differs from a standard synthesis process in that it takes into account
        where the circuits are (qargs) and respects it on the synthesized circuit.

        Args:
            circuits: A list of circuits to be synthesized.
            qargs: A list of lists of qubit indices for each circuit.
            coupling_map: A coupling map representing the connectivity.
            backend_name: The name of the backend (deprecated, not used).
            backend: The backend to use for the synthesis.

        Returns:
            List of synthesized QuantumCircuits or None for failed syntheses.
        """
        _ = backend_name  # Suppress unused warning, kept for API compatibility

        validate_coupling_map_source(coupling_map, backend)
        formatted_coupling_map = get_formatted_coupling_map(coupling_map)
        validate_circuits_and_qargs_lengths(circuits, qargs)

        coupling_map_graph = get_coupling_map_graph(backend, formatted_coupling_map)

        logger.info("Running %s AI synthesis on local mode", self.synthesis_type)

        return self._synthesize_circuits(coupling_map_graph, circuits, qargs)


class AILocalCliffordSynthesis(AILocalSynthesisBase):
    """Local-mode Clifford synthesis backed by cached RL models."""

    synthesis_type = "Clifford"

    def _prepare_input(
        self,
        circuit: QuantumCircuit | Clifford,
        subgraph_perm: list[int],
        target_qubits: int,
    ) -> Clifford | None:
        clifford = circuit if isinstance(circuit, Clifford) else Clifford(circuit)
        clifford = perm_cliff(clifford, subgraph_perm)

        if clifford.num_qubits > target_qubits:
            logger.warning(
                "Model expects %s qubits but circuit uses %s; skipping",
                target_qubits,
                clifford.num_qubits,
            )
            return None

        if clifford.num_qubits < target_qubits:
            clifford = embed_clifford(clifford, target_qubits)

        return clifford


class AILocalPauliNetworkSynthesis(AILocalSynthesisBase):
    """Local-mode Pauli network synthesis backed by cached RL models from qiskit-gym."""

    synthesis_type = "Pauli network"

    def _prepare_input(
        self,
        circuit: QuantumCircuit,
        subgraph_perm: list[int],
        target_qubits: int,
    ) -> QuantumCircuit | None:
        # Decompose certain gates that may not be directly supported
        input_circuit = circuit.decompose(
            ["swap", "rxx", "ryy", "rzz", "rzx", "rzy", "ryx"]
        )

        # Apply permutation to circuit
        input_circuit_perm = QuantumCircuit(input_circuit.num_qubits).compose(
            input_circuit, qubits=np.argsort(subgraph_perm)
        )

        if input_circuit_perm.num_qubits > target_qubits:
            logger.warning(
                "Model expects %s qubits but circuit uses %s; skipping",
                target_qubits,
                input_circuit_perm.num_qubits,
            )
            return None

        # Embed into larger qubit space if needed
        if input_circuit_perm.num_qubits < target_qubits:
            embedded = QuantumCircuit(target_qubits)
            embedded.compose(
                input_circuit_perm,
                qubits=range(input_circuit_perm.num_qubits),
                inplace=True,
            )
            return embedded

        return input_circuit_perm


class AILocalLinearFunctionSynthesis(AILocalSynthesisBase):
    """Local-mode linear-function synthesis backed by cached RL models."""

    synthesis_type = "Linear function"

    def _prepare_input(
        self,
        circuit: QuantumCircuit | LinearFunction,
        subgraph_perm: list[int],
        target_qubits: int,
    ) -> QuantumCircuit | None:
        if isinstance(circuit, LinearFunction):
            circuit_qc = QuantumCircuit(circuit.num_qubits)
            circuit_qc.append(circuit, range(circuit.num_qubits))
        else:
            circuit_qc = circuit

        clifford = Clifford(circuit_qc)
        clifford = perm_cliff(clifford, subgraph_perm)

        if clifford.num_qubits > target_qubits:
            logger.warning(
                "Model expects %s qubits but circuit uses %s; skipping",
                target_qubits,
                clifford.num_qubits,
            )
            return None

        if clifford.num_qubits < target_qubits:
            clifford = embed_clifford(clifford, target_qubits)

        return clifford.to_circuit()


class AILocalPermutationSynthesis(AILocalSynthesisBase):
    """Local-mode permutation synthesis backed by cached RL models."""

    synthesis_type = "Permutation"

    def _get_input_n_qubits(self, circuit: list[int]) -> int:
        """Get the number of qubits from a permutation list."""
        return len(circuit)

    def _embed_perm(self, perm_circ: list[int], num_qubits: int) -> list[int]:
        """Embed a smaller permutation array into a larger register."""
        if num_qubits < len(perm_circ):
            raise ValueError(
                f"Trying to embed a permutation with {len(perm_circ)} qubits "
                f"in a {num_qubits} qubits permutation."
            )
        new_perm_circ = list(range(num_qubits))
        new_perm_circ[: len(perm_circ)] = perm_circ
        return new_perm_circ

    def _prepare_input(
        self,
        circuit: list[int],
        subgraph_perm: list[int],
        target_qubits: int,
    ) -> list[int] | None:
        _ = subgraph_perm  # Not used for permutations, applied at output stage
        circ_n_qubits = len(circuit)

        if target_qubits < circ_n_qubits:
            logger.warning(
                "Model trained for %s qubits cannot synthesize a %s-qubit permutation",
                target_qubits,
                circ_n_qubits,
            )
            return None

        perm_input: list[int] = list(circuit)
        if target_qubits > circ_n_qubits:
            perm_input = self._embed_perm(perm_input, target_qubits)

        return perm_input
