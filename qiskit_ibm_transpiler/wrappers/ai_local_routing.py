# -*- coding: utf-8 -*-

# (C) Copyright 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit
from qiskit_ibm_transpiler.utils import get_circuit_from_qasm, input_to_qasm
from typing import List, Union, Literal
from qiskit_ibm_ai_local_transpiler import AIRoutingInference
from qiskit.transpiler import CouplingMap


# TODO: Reuse this code, it's repeated several times
OptimizationOptions = Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]


class AILocalRouting:
    """A helper class that covers the AILocalRouting funcionality"""

    def routing(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        optimization_level: int = 1,
        check_result: bool = False,
        layout_mode: str = "OPTIMIZE",
        optimization_preferences: Union[
            OptimizationOptions, List[OptimizationOptions], None
        ] = None,
    ):
        coupling_map_edges = list(coupling_map.get_edges())
        coupling_map_dists_array = coupling_map.distance_matrix.astype(int).tolist()
        coupling_map_n_qubits = len(coupling_map_dists_array)

        # Perform routing
        routed_qc, init_layout, final_layout = AIRoutingInference().route(
            circuit=circuit,
            coupling_map_edges=coupling_map_edges,
            coupling_map_n_qubits=coupling_map_n_qubits,
            coupling_map_dist_array=coupling_map_dists_array,
            layout_mode=layout_mode,
            op_params=optimization_level,
            optimization_preferences=optimization_preferences,
        )

        return routed_qc, init_layout, final_layout
