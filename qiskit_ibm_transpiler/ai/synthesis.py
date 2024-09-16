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

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Union

from qiskit.circuit.exceptions import CircuitError
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_ibm_transpiler.wrappers import (
    AICliffordAPI,
    AILinearFunctionAPI,
    AIPermutationAPI,
)

logger = logging.getLogger(__name__)

MAX_THREADS = os.environ.get("AI_TRANSPILER_MAX_THREADS", int(cpu_count() / 2))


class AISynthesis(TransformationPass):
    """AI Synthesis base class"""

    def __init__(
        self,
        synth_service: Union[AICliffordAPI, AILinearFunctionAPI, AIPermutationAPI],
        coupling_map: Union[List[List[int]], CouplingMap, None] = None,
        backend_name: Union[str, None] = None,
        replace_only_if_better: bool = True,
        max_threads: Union[int, None] = None,
        **kwargs,
    ) -> None:
        if isinstance(coupling_map, CouplingMap):
            self.coupling_map = list(coupling_map.get_edges())
        else:
            self.coupling_map = coupling_map
        self.backend_name = backend_name
        self.replace_only_if_better = replace_only_if_better
        self.synth_service = synth_service
        self.max_threads = max_threads if max_threads else MAX_THREADS
        super().__init__()

    def _should_keep_original(self, synth, original):
        if synth is None:
            return True
        return (
            original is not None
            and self.replace_only_if_better
            and self._is_original_a_better_circuit(synth, original)
        )

    def synth_nodes(self, nodes):
        if len(nodes) == 0:
            return [], []
        synth_inputs, originals = [], []
        try:
            for node in nodes:
                synth_inp, orig = self._get_synth_input_and_original(node)
                synth_inputs.append(synth_inp)
                originals.append(orig)
        except CircuitError:
            logger.warning(
                f"Error getting  synth input from node. Skipping ai transpilation."
            )
            return [], []

        try:
            qargs = [[q._index for q in node.qargs] for node in nodes]
            logger.debug(f"Attempting synthesis over qubits {qargs}")
            synths = self.synth_service.transpile(
                synth_inputs,
                qargs=qargs,
                coupling_map=self.coupling_map,
                backend_name=self.backend_name,
            )
        except TranspilerError as e:
            logger.warning(
                f"{self.synth_service.__class__.__name__} couldn't synthesize the circuit: {e}"
            )
            synths = [None] * len(synth_inputs)

        outputs = []
        for original, synth in zip(originals, synths):
            if self._should_keep_original(synth, original):
                logger.debug("Keeping the original circuit")
                output = original
            else:
                logger.debug("Using the synthesized circuit")
                output = synth
            outputs.append(output)

        return outputs, nodes

    def run(self, dag: DAGCircuit):
        logger.info(f"Requesting synthesis to the service")

        future_list = []

        blocks = [[] for _ in range(self.max_threads)]
        for i, node in enumerate(self._get_nodes(dag)):
            blocks[i % self.max_threads].append(node)

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            for block in blocks:
                future_list.append(executor.submit(self.synth_nodes, block))
            for future in as_completed(future_list):
                outputs, nodes = future.result()
                for output, node in zip(outputs, nodes):
                    if output:
                        dag.substitute_node_with_dag(node, circuit_to_dag(output))
        return dag


class AICliffordSynthesis(AISynthesis):
    """AICliffordSynthesis(backend_name: str, replace_only_if_better: bool = True, max_threads: int | None = None)

    Synthesis for `Clifford` circuits (blocks of `H`, `S` and `CX` gates). Currently up to 9 qubit blocks.

    :param backend_name: Name of the backend used for doing the AI Clifford synthesis.
    :type backend_name: str
    :param replace_only_if_better: Determine if replace the original circuit with the synthesized one if it's better, defaults to True.
    :type replace_only_if_better: bool, optional
    :param max_threads: Set the number of requests to send in parallel.
    :type max_threads: int, optional
    """

    def __init__(
        self,
        coupling_map: Union[List[List[int]], CouplingMap, None] = None,
        backend_name: Union[str, None] = None,
        replace_only_if_better: bool = True,
        max_threads: Union[int, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            AICliffordAPI(**kwargs),
            coupling_map,
            backend_name,
            replace_only_if_better,
            max_threads,
        )

    def _get_synth_input_and_original(self, node):
        if type(node.op) is Clifford:
            cliff, original = node.op, None
        else:
            cliff, original = node.op.params
        return cliff, original

    def _get_nodes(self, dag):
        return dag.named_nodes("clifford", "Clifford")

    def _is_original_a_better_circuit(self, synth, original):
        return (
            original.decompose("swap").num_nonlocal_gates()
            <= synth.num_nonlocal_gates()
        )


class AILinearFunctionSynthesis(AISynthesis):
    """AILinearFunctionSynthesis(backend_name: str, replace_only_if_better: bool = True, max_threads: int | None = None)

    Synthesis for `Linear Function` circuits (blocks of `CX` and `SWAP` gates). Currently up to 9 qubit blocks.

    :param backend_name: Name of the backend used for doing the AI Linear Function synthesis.
    :type backend_name: str
    :param replace_only_if_better: Determine if replace the original circuit with the synthesized one if it's better, defaults to True.
    :type replace_only_if_better: bool, optional
    :param max_threads: Set the number of requests to send in parallel.
    :type max_threads: int, optional
    """

    def __init__(
        self,
        coupling_map: Union[List[List[int]], CouplingMap, None] = None,
        backend_name: Union[str, None] = None,
        replace_only_if_better: bool = True,
        max_threads: Union[int, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            AILinearFunctionAPI(**kwargs),
            coupling_map,
            backend_name,
            replace_only_if_better,
            max_threads,
        )

    def _get_synth_input_and_original(self, node):
        return node.op, node.op.params[1]

    def _get_nodes(self, dag):
        return dag.named_nodes("linear_function", "Linear_function")

    def _is_original_a_better_circuit(self, synth, original):
        return (
            original.decompose("swap").num_nonlocal_gates()
            <= synth.num_nonlocal_gates()
        )


class AIPermutationSynthesis(AISynthesis):
    """AIPermutationSynthesis(backend_name: str, replace_only_if_better: bool = True, max_threads: int | None = None)

    Synthesis for `Permutation` circuits (blocks of `SWAP` gates). Currently available for 65, 33, and 27 qubit blocks.

    :param backend_name: Name of the backend used for doing the AI Linear Function synthesis.
    :type backend_name: str
    :param replace_only_if_better: Determine if replace the original circuit with the synthesized one if it's better, defaults to True.
    :type replace_only_if_better: bool, optional
    :param max_threads: Set the number of requests to send in parallel.
    :type max_threads: int, optional
    """

    def __init__(
        self,
        coupling_map: Union[List[List[int]], CouplingMap, None] = None,
        backend_name: Union[str, None] = None,
        replace_only_if_better: bool = True,
        max_threads: Union[int, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            AIPermutationAPI(**kwargs),
            coupling_map,
            backend_name,
            replace_only_if_better,
            max_threads,
        )

    def _get_synth_input_and_original(self, node):
        return node.op.params[0].tolist(), None

    def _get_nodes(self, dag):
        return dag.named_nodes("permutation", "Permutation")

    def _is_original_a_better_circuit(self, synth, original):
        return original.num_nonlocal_gates() <= synth.num_nonlocal_gates()
