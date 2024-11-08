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

from qiskit_ibm_ai_local_transpiler import AICliffordInference, AILinearFunctionInference, AIPermutationInference

from typing import Union, List

from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.providers.backend import BackendV2 as Backend
from qiskit.transpiler import CouplingMap

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AILocalSynthesis:
    """A helper class that covers some basic funcionality from the Linear Function AI Local Synthesis"""

    def __init__(self, inference_provider: Union[
            AICliffordInference,
            AILinearFunctionInference,
            AIPermutationInference,
        ]) -> None:
        self.inference_provider = inference_provider
        

    def transpile(
        self,
        circuits: List[Union[QuantumCircuit, LinearFunction]],
        qargs: List[List[int]],
        coupling_map: Union[List[List[int]], CouplingMap, None] = None,
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

        if not coupling_map and not backend:
            raise ValueError(
                "ERROR. Either a 'coupling_map' or a 'backend' must be provided."
            )

        if coupling_map:
            if isinstance(coupling_map, CouplingMap):
                formatted_coupling_map = coupling_map
            elif isinstance(coupling_map, list):
                formatted_coupling_map = CouplingMap(couplinglist=coupling_map)
            else:
                raise ValueError(
                    f"ERROR. coupling_map should either be a list of int tuples or a Qiskit CouplingMap object."
                )

        n_circs = len(circuits)
        n_qargs = len(qargs)

        if n_circs != n_qargs:
            raise ValueError(
                f"ERROR. The number of input circuits {n_circs}"
                f"and the number of input qargs arrays {n_qargs} are different."
            )

        logger.info("Running local AI synthesis on local mode")

        synthesized_circuits = []

        for index, circuit_qargs in enumerate(qargs):

            synthesized_linear_function = self.inference_provider().synthesize(
                circuit=circuits[index],
                coupling_map=formatted_coupling_map or backend.coupling_map,
                circuit_qargs=circuit_qargs,
                backend=backend,
            )

            synthesized_circuits.append(synthesized_linear_function)

        logger.info("Local AI synthesis on local mode completed")

        return synthesized_circuits
