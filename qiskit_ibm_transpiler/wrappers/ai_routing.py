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
from qiskit_ibm_transpiler.utils import (
    deserialize_circuit_from_qpy_or_qasm,
    get_circuit_from_qpy,
    get_qpy_from_circuit,
    serialize_circuit_to_qpy_or_qasm,
)
from .base import QiskitTranspilerService
from typing import List, Union, Literal


# TODO: Reuse this code, it's repeated several times
OptimizationOptions = Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]


class AIRoutingAPI(QiskitTranspilerService):
    """A helper class that covers some basic funcionality from the AIRouting API"""

    def __init__(self, **kwargs):
        super().__init__(path_param="routing", **kwargs)

    def routing(
        self,
        circuit: QuantumCircuit,
        coupling_map,
        optimization_level: int = 1,
        check_result: bool = False,
        layout_mode: str = "OPTIMIZE",
        optimization_preferences: Union[
            OptimizationOptions, List[OptimizationOptions], None
        ] = None,
    ):
        qpy, qasm = serialize_circuit_to_qpy_or_qasm(circuit)
        body_params = {
            "qasm": qasm,
            "qpy": qpy,
            "coupling_map": coupling_map,
            "optimization_preferences": optimization_preferences,
        }

        params = {
            "check_result": check_result,
            "layout_mode": layout_mode,
            "optimization_level": optimization_level,
        }

        routing_resp = self.request_and_wait(
            endpoint="routing", body=body_params, params=params
        )

        if routing_resp.get("success"):
            routed_circuit = deserialize_circuit_from_qpy_or_qasm(
                routing_resp["qpy"], routing_resp["qasm"]
            )
            return (
                routed_circuit,
                routing_resp["layout"]["initial"],
                routing_resp["layout"]["final"],
            )
