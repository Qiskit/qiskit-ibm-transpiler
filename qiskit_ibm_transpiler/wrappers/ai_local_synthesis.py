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

import importlib
import logging
from typing import List, Union

import networkx as nx
import numpy as np
from networkx.exception import NetworkXError
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.providers.backend import BackendV2 as Backend
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ai_local_package = "qiskit_ibm_ai_local_transpiler"
qiskit_ibm_ai_local_transpiler = (
    importlib.import_module(ai_local_package)
    if importlib.util.find_spec(ai_local_package)
    else None
)

AICliffordInference = getattr(
    qiskit_ibm_ai_local_transpiler,
    "AICliffordInference",
    "AICliffordInference not found",
)
AILinearFunctionInference = getattr(
    qiskit_ibm_ai_local_transpiler,
    "AILinearFunctionInference",
    "AILinearFunctionInference not found",
)
AIPermutationInference = getattr(
    qiskit_ibm_ai_local_transpiler,
    "AIPermutationInference",
    "AIPermutationInference not found",
)

qiskit_ibm_ai_local_transpiler_linear_function = getattr(
    qiskit_ibm_ai_local_transpiler,
    "linear_function",
    "linear_function module on qiskit_ibm_ai_local_transpiler not found",
)
qiskit_ibm_ai_local_transpiler_permutation = getattr(
    qiskit_ibm_ai_local_transpiler,
    "permutation",
    "permutation module on qiskit_ibm_ai_local_transpiler not found",
)
qiskit_ibm_ai_local_transpiler_clifford = getattr(
    qiskit_ibm_ai_local_transpiler,
    "clifford",
    "clifford module on qiskit_ibm_ai_local_transpiler not found",
)

LINEAR_FUNCTION_COUPLING_MAPS_BY_HASHES_DICT = getattr(
    qiskit_ibm_ai_local_transpiler_linear_function,
    "LINEAR_FUNCTION_COUPLING_MAPS_BY_HASHES_DICT",
    "LINEAR_FUNCTION_COUPLING_MAPS_BY_HASHES_DICT not found",
)
PERMUTATION_COUPLING_MAPS_BY_HASHES_DICT = getattr(
    qiskit_ibm_ai_local_transpiler_permutation,
    "PERMUTATION_COUPLING_MAPS_BY_HASHES_DICT",
    "PERMUTATION_COUPLING_MAPS_BY_HASHES_DICT not found",
)
CLIFFORD_COUPLING_MAPS_BY_HASHES_DICT = getattr(
    qiskit_ibm_ai_local_transpiler_clifford,
    "CLIFFORD_COUPLING_MAPS_BY_HASHES_DICT",
    "CLIFFORD_COUPLING_MAPS_BY_HASHES_DICT not found",
)


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
    coupling_map: nx.Graph, circuit_qargs: List[int], coupling_maps_by_hashes: dict
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


def get_synthesized_linear_function_circuits(
    coupling_map: nx.Graph, clifford_dicts: List[dict], qargs: List[List[int]]
) -> list[QuantumCircuit]:
    synthesized_circuits = []

    for index, circuit_qargs in enumerate(qargs):
        try:
            subgraph_perm, cmap_hash = get_mapping_perm(
                coupling_map,
                circuit_qargs,
                LINEAR_FUNCTION_COUPLING_MAPS_BY_HASHES_DICT,
            )
        except BaseException as e:
            logger.warning(e)
            continue

        # Generate the Clifford from the dictionary to send it to the model and permute it
        clifford = perm_cliff(Clifford.from_dict(clifford_dicts[index]), subgraph_perm)

        synthesized_linear_function = AILinearFunctionInference().synthesize(
            cliff=clifford, coupling_map_hash=cmap_hash
        )

        # Permute the circuit back
        synthesized_circuit = QuantumCircuit(
            synthesized_linear_function.num_qubits
        ).compose(synthesized_linear_function, qubits=subgraph_perm)

        # synthesized_circuit could be None or have a value, we return it in both cases
        synthesized_circuits.append(synthesized_circuit)

    return synthesized_circuits


def get_synthesized_clifford_circuits(
    coupling_map: nx.Graph, clifford_dicts: List[dict], qargs: List[List[int]]
) -> list[QuantumCircuit]:
    synthesized_circuits = []

    for index, circuit_qargs in enumerate(qargs):
        try:
            subgraph_perm, cmap_hash = get_mapping_perm(
                coupling_map,
                circuit_qargs,
                CLIFFORD_COUPLING_MAPS_BY_HASHES_DICT,
            )
        except BaseException as e:
            logger.warning(e)
            continue

        # Generate the Clifford from the dictionary to send it to the model and permute it
        clifford = perm_cliff(Clifford.from_dict(clifford_dicts[index]), subgraph_perm)

        synthesized_linear_function = AICliffordInference().synthesize(
            cliff=clifford, coupling_map_hash=cmap_hash
        )

        # Permute the circuit back
        synthesized_circuit = QuantumCircuit(
            synthesized_linear_function.num_qubits
        ).compose(synthesized_linear_function, qubits=subgraph_perm)

        # synthesized_circuit could be None or have a value, we return it in both cases
        synthesized_circuits.append(synthesized_circuit)

    return synthesized_circuits


class AILocalCliffordSynthesis:
    """A helper class that covers some basic funcionality from the Linear Function AI Local Synthesis"""

    def transpile(
        self,
        circuits: List[Union[QuantumCircuit, Clifford]],
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], CouplingMap, None] = None,
        # backend_name is not used here but is maintained until we deprecate it to not break the code
        backend_name=None,
        backend: Union[Backend, None] = None,
    ) -> List[Union[QuantumCircuit, None]]:
        """Synthetize one or more quantum circuits into an optimized equivalent. It differs from a standard synthesis process in that it takes into account where the linear functions are (qargs)
        and respects it on the synthesized circuit.

        Args:
            circuits (List[Union[QuantumCircuit, Clifford]]): A list of quantum circuits to be synthesized.
            qargs (List[List[int]]): A list of lists of qubit indices for each circuit. Each list of qubits indices represent where the linear function circuit is.
            coupling_map (Union[List[List[int]], None]): A coupling map representing the connectivity of the quantum computer.
            backend_name (Union[str, None]): The name of the backend to use for the synthesis.

        Returns:
            List[Union[QuantumCircuit, None]]: A list of synthesized quantum circuits. If the synthesis fails for any circuit, the corresponding element in the list will be None.
        """

        # Although this function is called `transpile`, it does a synthesis. It has this name because the synthesis
        # is made as a pass on the Qiskit Pass Manager which is used in the transpilation process.

        validate_coupling_map_source(coupling_map, backend)

        formatted_coupling_map = get_formatted_coupling_map(coupling_map)

        validate_circuits_and_qargs_lengths(circuits, qargs)

        clifford_dict = [Clifford(circuit).to_dict() for circuit in circuits]

        coupling_map_graph = get_coupling_map_graph(backend, formatted_coupling_map)

        logger.info("Running Clifford AI synthesis on local mode")

        synthesized_circuits = get_synthesized_clifford_circuits(
            coupling_map_graph, clifford_dict, qargs
        )

        return synthesized_circuits


class AILocalLinearFunctionSynthesis:
    """A helper class that covers some basic funcionality from the Linear Function AI Local Synthesis"""

    def transpile(
        self,
        circuits: List[Union[QuantumCircuit, LinearFunction]],
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], CouplingMap, None] = None,
        # backend_name is not used here but is maintained until we deprecate it to not break the code
        backend_name=None,
        backend: Union[Backend, None] = None,
    ) -> List[Union[QuantumCircuit, None]]:
        """Synthetize one or more quantum circuits into an optimized equivalent. It differs from a standard synthesis process in that it takes into account where the linear functions are (qargs)
        and respects it on the synthesized circuit.

        Args:
            circuits (List[Union[QuantumCircuit, LinearFunction]]): A list of quantum circuits to be synthesized.
            qargs (List[List[int]]): A list of lists of qubit indices for each circuit. Each list of qubits indices represent where the linear function circuit is.
            coupling_map (Union[List[List[int]], None]): A coupling map representing the connectivity of the quantum computer.
            backend_name (Union[str, None]): The name of the backend to use for the synthesis.

        Returns:
            List[Union[QuantumCircuit, None]]: A list of synthesized quantum circuits. If the synthesis fails for any circuit, the corresponding element in the list will be None.
        """

        # Although this function is called `transpile`, it does a synthesis. It has this name because the synthesis
        # is made as a pass on the Qiskit Pass Manager which is used in the transpilation process.

        validate_coupling_map_source(coupling_map, backend)

        formatted_coupling_map = get_formatted_coupling_map(coupling_map)

        validate_circuits_and_qargs_lengths(circuits, qargs)

        clifford_dict = [Clifford(circuit).to_dict() for circuit in circuits]

        coupling_map_graph = get_coupling_map_graph(backend, formatted_coupling_map)

        logger.info("Running Linear Functions AI synthesis on local mode")

        synthesized_circuits = get_synthesized_linear_function_circuits(
            coupling_map_graph, clifford_dict, qargs
        )

        return synthesized_circuits


def get_synthesized_permutation_circuits(
    coupling_map: nx.Graph, permutations_list: List[List[int]], qargs: List[List[int]]
) -> list[QuantumCircuit]:
    synthesized_circuits = []

    for permutation, circuit_qargs in zip(permutations_list, qargs):
        try:
            subgraph_perm, cmap_hash = get_mapping_perm(
                coupling_map, circuit_qargs, PERMUTATION_COUPLING_MAPS_BY_HASHES_DICT
            )
        except BaseException as e:
            logger.warning(e)
            continue

        synthesized_permutation = AIPermutationInference().synthesize(
            perm_circ=permutation, coupling_map_hash=cmap_hash
        )

        # Permute the circuit back
        synthesized_circuit = QuantumCircuit(
            synthesized_permutation.num_qubits
        ).compose(synthesized_permutation, qubits=subgraph_perm)

        # synthesized_circuit could be None or have a value, we return it in both cases
        synthesized_circuits.append(synthesized_circuit)

    return synthesized_circuits


class AILocalPermutationSynthesis:
    """A helper class that covers some basic funcionality from the Permutation AI Local Synthesis"""

    def transpile(
        self,
        circuits: List[List[int]],
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], CouplingMap, None] = None,
        # backend_name is not used here but is maintained until we deprecate it to not break the code
        backend_name=None,
        backend: Union[Backend, None] = None,
    ) -> List[Union[QuantumCircuit, None]]:
        """Synthetize one or more quantum circuits into an optimized equivalent. It differs from a standard synthesis process in that it takes into account where the Permutations are (qargs)
        and respects it on the synthesized circuit.

        Args:
            circuits (List[List[int]]): A list of quantum circuits to be synthesized.
            qargs (List[List[int]]): A list of lists of qubit indices for each circuit. Each list of qubits indices represent where the Permutation circuit is.
            coupling_map (Union[List[List[int]], None]): A coupling map representing the connectivity of the quantum computer.
            backend_name (Union[str, None]): The name of the backend to use for the synthesis.

        Returns:
            List[Union[QuantumCircuit, None]]: A list of synthesized quantum circuits. If the synthesis fails for any circuit, the corresponding element in the list will be None.
        """

        # Although this function is called `transpile`, it does a synthesis. It has this name because the synthesis
        # is made as a pass on the Qiskit Pass Manager which is used in the transpilation process.

        validate_coupling_map_source(coupling_map, backend)

        formatted_coupling_map = get_formatted_coupling_map(coupling_map)

        validate_circuits_and_qargs_lengths(circuits, qargs)

        coupling_map_graph = get_coupling_map_graph(backend, formatted_coupling_map)

        logger.info("Running Permutations AI synthesis on local mode")

        synthesized_circuits = get_synthesized_permutation_circuits(
            coupling_map_graph, circuits, qargs
        )

        return synthesized_circuits
