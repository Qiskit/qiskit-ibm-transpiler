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
import networkx as nx
from networkx.exception import NetworkXError

from qiskit_ibm_transpiler.ai.rl_inferences.linear_functions import (
    LinearFunctionInference,
)

from typing import Union, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library import LinearFunction
from qiskit.quantum_info import Clifford
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_transpiler.ai.models.linear_functions import (
    model_coupling_map_by_model_hash,
    MODEL_HASHES as MODEL_LIN_FUNC_HASHES,
)
from qiskit_ibm_transpiler.utils import get_qasm_from_circuit

logger = logging.getLogger(__name__)


class AILocalLinearFunctionSynthesis:
    """A helper class that covers some basic funcionality from the Linear Function AI Local Synthesis"""

    def transpile(
        self,
        circuits: List[Union[QuantumCircuit, LinearFunction]],
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], None] = None,
        backend_name: Union[str, None] = None,
    ) -> List[Union[QuantumCircuit, None]]:
        """Synthetize one or more quantum circuits into an optimized equivalent. It differs from a standard synthesis process in that it takes into account where the linear functions are (qargs)
        and respects it on the synthetized circuit.

        Args:
            circuits (List[Union[QuantumCircuit, LinearFunction]]): A list of quantum circuits to be synthetized.
            qargs (List[List[int]]): A list of lists of qubit indices for each circuit. Each list of qubits indices represent where the linear function circuit is.
            coupling_map (Union[List[List[int]], None]): A coupling map representing the connectivity of the quantum computer.
            backend_name (Union[str, None]): The name of the backend to use for the synthesis.

        Returns:
            List[Union[QuantumCircuit, None]]: A list of synthetized quantum circuits. If the synthesis fails for any circuit, the corresponding element in the list will be None.
        """

        # Although this function is called `transpile`, it does a synthesis. It has this name because the synthesis
        # is made as a pass on the Qiskit Pass Manager which is used in the transpilation process.

        if not coupling_map and not backend_name:
            raise ValueError(
                f"ERROR. Either a 'coupling_map' or a 'backend_name' must be provided."
            )

        n_circs = len(circuits)
        n_qargs = len(qargs)

        if n_circs != n_qargs:
            raise ValueError(
                f"ERROR. The number of input circuits {n_circs}"
                f"and the number of input qargs arrays {n_qargs} are different."
            )

        clifford_dict = [Clifford(circuit).to_dict() for circuit in circuits]

        coupling_map_graph = get_coupling_map_graph(backend_name, coupling_map)

        synthetized_circuits = synthetize_linear_functions(
            coupling_map_graph, clifford_dict, qargs
        )

        return synthetized_circuits


def perm_cliff(cliff, perm):
    perm = np.array(perm)
    cliff.stab_x = cliff.stab_x[:, perm]
    cliff.stab_z = cliff.stab_z[:, perm]
    cliff.destab_x = cliff.destab_x[:, perm]
    cliff.destab_z = cliff.destab_z[:, perm]
    cliff.stab = cliff.stab[perm, :]
    cliff.destab = cliff.destab[perm, :]
    return cliff


def synthetize_linear_functions(
    coupling_map: nx.Graph, clifford_dict, qargs: List[List[int]]
):
    synthesis_response = []

    for index, circuit_qargs in enumerate(qargs):
        try:
            subgraph_perm, cmap_hash = get_mapping_perm(coupling_map, circuit_qargs)
        except BaseException:
            raise AttributeError(f"ERROR. Malformed qargs {circuit_qargs}")

        # Generate the Clifford from the dictionary to send it to the model and permute it
        clifford = perm_cliff(Clifford.from_dict(clifford_dict[index]), subgraph_perm)

        synthetized_circuit = LinearFunctionInference().synthesize(
            cliff=clifford, coupling_map_hash=cmap_hash
        )
        # Permute the circuit back
        synthetized_circuit = QuantumCircuit(synthetized_circuit.num_qubits).compose(
            synthetized_circuit, qubits=subgraph_perm
        )

        transpilation_succeded = False if synthetized_circuit is None else True
        qasm_circuit = get_qasm_from_circuit(synthetized_circuit)

        synthetized_circuit_formatted = None
        if transpilation_succeded and qasm_circuit:
            synthetized_circuit_formatted = QuantumCircuit.from_qasm_str(qasm_circuit)

        synthesis_response.append(synthetized_circuit_formatted)

    return synthesis_response


def get_coupling_map_graph(
    backend_name: str | None = None, backend_coupling_map: list[list[int]] | None = None
) -> nx.Graph:
    coupling_map_list_format = None

    if backend_coupling_map:
        coupling_map_list_format = backend_coupling_map
    elif backend_name:
        try:
            runtime_service = QiskitRuntimeService()
            backend_info = runtime_service.backend(name=backend_name)
            coupling_map_edges = CouplingMap.get_edges(backend_info.coupling_map)
            coupling_map_list_format = [list(edge) for edge in coupling_map_edges]
        except Exception:
            raise PermissionError(f"ERROR. Backend not supported ({backend_name})")

    try:
        coupling_map = nx.Graph(coupling_map_list_format)
    except Exception:
        raise NetworkXError(f"ERROR. Cannot convert coupling_map from list to graph")

    return coupling_map


def get_mapping_perm(coupling_map: nx.Graph, circuit_qargs: List[int]) -> list[int]:
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
    if circuit_in_coupling_map_hash not in MODEL_LIN_FUNC_HASHES:
        raise LookupError(f"ERROR. No model available for the requested subgraph")

    model_coupling_map = model_coupling_map_by_model_hash[circuit_in_coupling_map_hash]

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
