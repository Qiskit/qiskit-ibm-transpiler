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

import os
from urllib.parse import urljoin

from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.qasm2 import QASM2ExportError

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
        is_qasm3 = False
        try:
            qasm = qasm2.dumps(circuit)
        except QASM2ExportError:
            qasm = qasm3.dumps(circuit)
            is_qasm3 = True

        body_params = {
            "qasm": qasm.replace("\n", " "),
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
            routed_circuit = (
                qasm3.loads(routing_resp["qasm"])
                if is_qasm3
                else QuantumCircuit.from_qasm_str(routing_resp["qasm"])
            )
            return (
                routed_circuit,
                routing_resp["layout"]["initial"],
                routing_resp["layout"]["final"],
            )
