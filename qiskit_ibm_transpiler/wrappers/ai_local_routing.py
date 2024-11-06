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
import logging

logger = logging.getLogger(__name__)


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
        optimization_level: dict | int = 1,
        check_result: bool = False,
        layout_mode: str = "optimize",
        optimization_preferences: Union[
            OptimizationOptions, List[OptimizationOptions], None
        ] = None,
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
        logger.debug("Running local routing inference")
        routed_qc, init_layout, final_layout = AIRoutingInference().route(
            circuit=circuit,
            coupling_map_edges=coupling_map_edges,
            coupling_map_n_qubits=coupling_map_n_qubits,
            coupling_map_dist_array=coupling_map_dists_array,
            layout_mode=layout_mode.lower(),
            op_params=op_params,
            optimization_preferences=optimization_preferences,
        )
        logger.debug("Local routing inference completed")

        return routed_qc, init_layout, final_layout
