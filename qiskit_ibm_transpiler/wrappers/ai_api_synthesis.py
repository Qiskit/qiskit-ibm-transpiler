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
from typing import List, Union

from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.providers.backend import BackendV2 as Backend
from qiskit.quantum_info import Clifford

from ..utils import (
    serialize_circuits_to_qpy_or_qasm,
)
from .base import QiskitTranspilerService

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
    ) -> List[Union[QuantumCircuit, None]]:
        """Synthetize one or more quantum circuits into an optimized equivalent. It differs from a standard synthesis process in that it takes into account where the cliffords are (qargs)
        and respects it on the synthesized circuit.

        Args:
            circuits (List[Union[QuantumCircuit, Clifford]]): A list of quantum circuits to be synthesized.
            qargs (List[List[int]]): A list of lists of qubit indices for each circuit. Each list of qubits indices represent where the cliffors circuit is.
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

        if coupling_map is not None:
            body_params["backend_coupling_map"] = coupling_map
        elif backend_name is not None:
            query_params["backend"] = backend_name

        logger.info("Running synthesis against the Qiskit Transpiler Service")

        transpile_response = self.request_and_wait(
            endpoint="transpile",
            body=body_params,
            params=query_params,
        )

        synthesized_circuits = self._handle_response(transpile_response)

        return synthesized_circuits


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

        synthesized_circuits = self._handle_response(transpile_response)

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
    ) -> List[Union[QuantumCircuit, None]]:
        """Synthetize one or more permutation arrays into an optimized circuit equivalent. It differs from a standard synthesis process in that it takes into account where the permutations are (qargs)
        and respects it on the synthesized circuit.

        Args:
            patterns: List[List[int]]: A list of permutation arrays to be synthesized.
            qargs (List[List[int]]): A list of lists of qubit indices for each permutation array. Each list of qubits indices represent where the permutation array is.
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
            "permutation": patterns,
            "qargs": qargs,
        }

        query_params = dict()

        if coupling_map is not None:
            body_params["backend_coupling_map"] = coupling_map
        elif backend_name is not None:
            query_params["backend"] = backend_name

        logger.info("Running synthesis against the Qiskit Transpiler Service")

        transpile_response = self.request_and_wait(
            endpoint="transpile",
            body=body_params,
            params=query_params,
        )

        synthesized_circuits = self._handle_response(transpile_response)

        return synthesized_circuits


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
        if not coupling_map and not backend_name:
            raise ValueError(
                "ERROR. Either a 'coupling_map' or a 'backend_name' must be provided."
            )

        qpy, qasm = serialize_circuits_to_qpy_or_qasm(
            circuits, self.get_qiskit_version()
        )
        body_params = {
            "qasm": qasm,
            "qpy": qpy,
            "qargs": qargs,
        }

        query_params = dict()

        if coupling_map is not None:
            body_params["backend_coupling_map"] = coupling_map
        elif backend_name is not None:
            query_params["backend"] = backend_name

        logger.info("Running synthesis against the Qiskit Transpiler Service")

        transpile_response = self.request_and_wait(
            endpoint="transpile",
            body=body_params,
            params=query_params,
        )

        synthesized_circuits = self._handle_response(transpile_response)

        return synthesized_circuits
