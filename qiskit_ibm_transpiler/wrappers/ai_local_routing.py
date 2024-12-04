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

import importlib
from typing import List, Literal, Union

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from qiskit_ibm_transpiler.utils import get_circuit_from_qasm, input_to_qasm

ai_local_package = "qiskit_ibm_ai_local_transpiler"
qiskit_ibm_ai_local_transpiler = (
    importlib.import_module(ai_local_package)
    if importlib.util.find_spec(ai_local_package)
    else None
)
AIRoutingInference = getattr(
    qiskit_ibm_ai_local_transpiler, "AIRoutingInference", "AIRoutingInference not found"
)


# TODO: Reuse this code, it's repeated several times
OptimizationOptions = Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]

OP_LEVELS = {
    1: {"full_its": 8, "its": 2, "reps": 2, "runs": 1, "max_time": 30},
    2: {"full_its": 16, "its": 16, "reps": 8, "runs": 1, "max_time": 30},
    3: {"full_its": 32, "its": 16, "reps": 8, "runs": 1, "max_time": 300},
    100: {"full_its": 32, "its": 32, "reps": 32, "runs": 1, "max_time": 3000},
}


class AILocalRouting:
    """A helper class that covers the AILocalRouting funcionality"""

    def routing(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        optimization_level: Union[dict, int] = 1,
        check_result: bool = False,
        layout_mode: str = "optimize",
        optimization_preferences: List[OptimizationOptions] = None,
    ):
        coupling_map_edges = list(coupling_map.get_edges())
        coupling_map_dists_array = coupling_map.distance_matrix.astype(int).tolist()
        coupling_map_n_qubits = len(coupling_map_dists_array)

        op_params = OP_LEVELS[optimization_level]

        if type(optimization_level) is dict:
            # Users can provide their own values by providing a dict
            op_params = OP_LEVELS[3].copy()
            op_params.update(optimization_level)

        # Perform routing
        routed_qc, init_layout, final_layout = AIRoutingInference().route(
            circuit=circuit,
            coupling_map_edges=coupling_map_edges,
            coupling_map_n_qubits=coupling_map_n_qubits,
            coupling_map_dist_array=coupling_map_dists_array,
            layout_mode=layout_mode.lower(),
            op_params=op_params,
            optimization_preferences=optimization_preferences,
        )

        return routed_qc, init_layout, final_layout
