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
from qiskit.providers.backend import BackendV2 as Backend

from .base import QiskitTranspilerService
from ..utils import (
    serialize_circuits_to_qpy_or_qasm,
    deserialize_circuit_from_qpy_or_qasm,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AICliffordAPI(QiskitTranspilerService):
    """A helper class that covers some basic funcionality from the Clifford AI Synthesis API"""

    def __init__(self, **kwargs):
        super().__init__(path_param="clifford", **kwargs)

    def transpile(
        self,
        circuits: List[Union[QuantumCircuit, Clifford]],
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], None] = None,
        backend_name: Union[str, None] = None,
        # backend is not used yet, but probably it will replace backend_name
        backend: Union[Backend, None] = None,
    ):
        if coupling_map is not None:
            transpile_resps = self.request_and_wait(
                endpoint="transpile",
                body={
                    "clifford_dict": [
                        Clifford(circuit).to_dict() for circuit in circuits
                    ],
                    "qargs": qargs,
                    "backend_coupling_map": coupling_map,
                },
                params=dict(),
            )
        elif backend_name is not None:
            transpile_resps = self.request_and_wait(
                endpoint="transpile",
                body={
                    "clifford_dict": [
                        Clifford(circuit).to_dict() for circuit in circuits
                    ],
                    "qargs": qargs,
                },
                params={"backend": backend_name},
            )
        else:
            raise ValueError(
                "ERROR. Either a 'coupling_map' or a 'backend_name' must be provided."
            )

        results = []
        for transpile_resp in transpile_resps:
            if transpile_resp.get("success"):
                results.append(
                    deserialize_circuit_from_qpy_or_qasm(
                        transpile_resp.get("qpy"), transpile_resp.get("qasm")
                    )
                )
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
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], None] = None,
        backend_name: Union[str, None] = None,
        # backend is not used yet, but probably it will replace backend_name
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

        if not coupling_map and not backend_name:
            raise ValueError(
                "ERROR. Either a 'coupling_map' or a 'backend_name' must be provided."
            )

        body_params = {
            "clifford_dict": [Clifford(circuit).to_dict() for circuit in circuits],
            "qargs": qargs,
        }

        query_params = dict()

        if coupling_map:
            body_params["backend_coupling_map"] = coupling_map
        elif backend_name:
            query_params["backend"] = backend_name

        logger.info("Running synthesis against the Qiskit Transpiler Service")

        transpile_response = self.request_and_wait(
            endpoint="transpile",
            body=body_params,
            params=query_params,
        )

        synthesized_circuits = []
        for response_element in transpile_response:
            synthesized_circuit = None
            if response_element.get("success"):
                synthesized_circuit = deserialize_circuit_from_qpy_or_qasm(
                    response_element.get("qpy"), response_element.get("qasm")
                )
            synthesized_circuits.append(synthesized_circuit)

        return synthesized_circuits


class AIPermutationAPI(QiskitTranspilerService):
    """A helper class that covers some basic funcionality from the Permutation AI Synthesis API"""

    def __init__(self, **kwargs):
        super().__init__(path_param="permutations", **kwargs)

    def transpile(
        self,
        patterns: List[List[int]],
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], None] = None,
        backend_name: Union[str, None] = None,
        # backend is not used yet, but probably it will replace backend_name
        backend: Union[Backend, None] = None,
    ):

        if coupling_map is not None:
            transpile_resps = self.request_and_wait(
                endpoint="transpile",
                body={
                    "permutation": patterns,
                    "qargs": qargs,
                    "backend_coupling_map": coupling_map,
                },
                params=dict(),
            )
        elif backend_name is not None:
            transpile_resps = self.request_and_wait(
                endpoint="transpile",
                body={
                    "permutation": patterns,
                    "qargs": qargs,
                },
                params={"backend": backend_name},
            )
        else:
            raise ValueError(
                "ERROR. Either a 'coupling_map' or a 'backend_name' must be provided."
            )

        results = []
        for transpile_resp in transpile_resps:
            if transpile_resp.get("success"):
                results.append(
                    deserialize_circuit_from_qpy_or_qasm(
                        transpile_resp.get("qpy"), transpile_resp.get("qasm")
                    )
                )
            else:
                results.append(None)
        return results


class AIPauliNetworkAPI(QiskitTranspilerService):
    """A helper class that covers some basic funcionality from the Pauli Network AI Synthesis API"""

    def __init__(self, **kwargs):
        super().__init__(path_param="pauli_network", **kwargs)

    def transpile(
        self,
        circuits: List[QuantumCircuit],
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], None] = None,
        backend_name: Union[str, None] = None,
        # backend is not used yet, but probably it will replace backend_name
        backend: Union[Backend, None] = None,
    ):
        qpy, qasm = serialize_circuits_to_qpy_or_qasm(circuits)
        if coupling_map is not None:
            transpile_resps = self.request_and_wait(
                endpoint="transpile",
                body={
                    "qasm": qasm,
                    "qpy": qpy,
                    "qargs": qargs,
                    "backend_coupling_map": coupling_map,
                },
                params={"backend": ""},
            )
        elif backend_name is not None:
            transpile_resps = self.request_and_wait(
                endpoint="transpile",
                body={
                    "qasm": qasm,
                    "qpy": qpy,
                    "qargs": qargs,
                },
                params={"backend": backend_name},
            )
        else:
            raise ValueError(
                f"ERROR. Either a 'coupling_map' or a 'backend_name' must be provided."
            )

        results = []
        for transpile_resp in transpile_resps:
            if transpile_resp.get("success"):
                results.append(
                    deserialize_circuit_from_qpy_or_qasm(
                        transpile_resp.get("qpy"), transpile_resp.get("qasm")
                    )
                )
            else:
                results.append(None)
        return results
