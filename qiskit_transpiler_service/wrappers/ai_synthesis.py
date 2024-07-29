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
from typing import Union, List

from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.quantum_info import Clifford

from .base import QiskitTranspilerService

logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.INFO)


class AICliffordAPI(QiskitTranspilerService):
    """A helper class that covers some basic funcionality from the Clifford AI Synthesis API"""

    def __init__(self, **kwargs):
        super().__init__(path_param="clifford", **kwargs)

    def transpile(
        self,
        circuits: List[Union[QuantumCircuit, Clifford]],
        backend: str,
        qargs: List[List[int]],
    ):
        transpile_resps = self.request_and_wait(
            endpoint="transpile",
            body={
                "clifford_dict": [Clifford(circuit).to_dict() for circuit in circuits],
                "qargs": qargs,
            },
            params={"backend": backend},
        )

        results = []
        for transpile_resp in transpile_resps:
            if transpile_resp.get("success") and transpile_resp.get("qasm") is not None:
                results.append(QuantumCircuit.from_qasm_str(transpile_resp.get("qasm")))
            else:
                results.append(None)
        return results


class AILinearFunctionAPI(QiskitTranspilerService):
    """A helper class that covers some basic funcionality from the Linear Function AI Synthesis API"""

    def __init__(self, **kwargs):
        super().__init__(path_param="linear_functions", **kwargs)

    def transpile(
        self,
        circuits: List[Union[QuantumCircuit, LinearFunction]],
        backend: str,
        qargs: List[List[int]],
    ):
        transpile_resps = self.request_and_wait(
            endpoint="transpile",
            body={
                "clifford_dict": [Clifford(circuit).to_dict() for circuit in circuits],
                "qargs": qargs,
            },
            params={"backend": backend},
        )

        results = []
        for transpile_resp in transpile_resps:
            if transpile_resp.get("success") and transpile_resp.get("qasm") is not None:
                results.append(QuantumCircuit.from_qasm_str(transpile_resp.get("qasm")))
            else:
                results.append(None)
        return results


class AIPermutationAPI(QiskitTranspilerService):
    """A helper class that covers some basic funcionality from the Permutation AI Synthesis API"""

    def __init__(self, **kwargs):
        super().__init__(path_param="permutations", **kwargs)

    def transpile(
        self,
        patterns: List[List[int]],
        backend: str,
        qargs: List[List[int]],
    ):
        transpile_resps = self.request_and_wait(
            endpoint="transpile",
            body={"permutation": patterns, "qargs": qargs},
            params={"backend": backend},
        )

        results = []
        for transpile_resp in transpile_resps:
            if transpile_resp.get("success") and transpile_resp.get("qasm") is not None:
                results.append(QuantumCircuit.from_qasm_str(transpile_resp.get("qasm")))
            else:
                results.append(None)
        return results
