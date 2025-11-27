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
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import PermutationGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.providers.backend import BackendV2 as Backend
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_ibm_transpiler.wrappers import (
    AICliffordAPI,
    AILinearFunctionAPI,
    AIPauliNetworkAPI,
    AIPermutationAPI,
)
from qiskit_ibm_transpiler.wrappers.ai_local_synthesis import (
    AILocalCliffordSynthesis,
    AILocalLinearFunctionSynthesis,
    AILocalPauliNetworkSynthesis,
    AILocalPermutationSynthesis,
)

logger = logging.getLogger(__name__)

MAX_THREADS = os.environ.get("AI_TRANSPILER_MAX_THREADS", int(cpu_count() / 2))


class AISynthesis(TransformationPass):
    """AI Synthesis base class"""

    def __init__(
        self,
        synth_service: (
            AICliffordAPI
            | AILinearFunctionAPI
            | AIPermutationAPI
            | AIPauliNetworkAPI
            | AILocalCliffordSynthesis
            | AILocalLinearFunctionSynthesis
            | AILocalPermutationSynthesis
            | AILocalPauliNetworkSynthesis
        ),
        coupling_map: list[list[int]] | CouplingMap | None = None,
        backend: Backend | None = None,
        replace_only_if_better: bool = True,
        max_threads: int | None = None,
        local_mode: bool = True,
        **kwargs,
    ) -> None:
        ai_local_package = "qiskit_ibm_ai_local_transpiler"
        if local_mode:
            if importlib.util.find_spec(ai_local_package) is None:
                raise ImportError(
                    f"For using the local mode you need to install the package '{ai_local_package}'. Read the installation guide for more information"
                )

        if backend and coupling_map:
            raise ValueError(
                f"ERROR. Both backend and coupling_map were specified as options. Please just use one of them."
            )

        if not backend and not coupling_map:
            raise ValueError(
                f"ERROR. One of these options must be set: backend or coupling_map."
            )

        if coupling_map:
            if isinstance(coupling_map, CouplingMap):
                self.coupling_map = coupling_map
            elif isinstance(coupling_map, list):
                self.coupling_map = CouplingMap(coupling_map)
            else:
                raise ValueError(
                    f"ERROR. coupling_map should either be a list of int tuples or a Qiskit CouplingMap object."
                )

        if backend:
            self.backend = backend
            self.backend_name = backend.name

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

            coupling_map = getattr(self, "coupling_map", None)
            if isinstance(coupling_map, CouplingMap):
                coupling_map = list(coupling_map.get_edges())

            synths = self.synth_service.transpile(
                synth_inputs,
                qargs=qargs,
                coupling_map=coupling_map,
                backend=getattr(self, "backend", None),
                backend_name=getattr(self, "backend_name", None),
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
                outputs.append(original)
            else:
                logger.debug("Using the synthesized circuit")
                outputs.append(synth)

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
    """AICliffordSynthesis(coupling_map: list[list[int]] | None = None, backend: Backend | None = None, replace_only_if_better: bool = True, max_threads: int | None = None)

    Synthesis for `Clifford` circuits (blocks of `H`, `S` and `CX` gates). Currently up to 9 qubit blocks.

    :param coupling_map: A list of pairs that represents physical links between qubits.
    :type coupling_map: list[list[int]], optional
    :param backend: Backend used for doing the AI Clifford synthesis.
    :type backend: Backend, optional
    :param replace_only_if_better: Determine if replace the original circuit with the synthesized one if it's better, defaults to True.
    :type replace_only_if_better: bool, optional
    :param max_threads: Set the number of requests to send in parallel.
    :type max_threads: int, optional
    """

    def __init__(
        self,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        backend: Backend | None = None,
        replace_only_if_better: bool = True,
        max_threads: int | None = None,
        local_mode: bool = True,
        **kwargs,
    ) -> None:
        ai_synthesis_provider = (
            AILocalCliffordSynthesis() if local_mode else AICliffordAPI(**kwargs)
        )

        super().__init__(
            synth_service=ai_synthesis_provider,
            coupling_map=coupling_map,
            backend=backend,
            replace_only_if_better=replace_only_if_better,
            max_threads=max_threads,
            local_mode=local_mode,
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
    """AILinearFunctionSynthesis(coupling_map: list[list[int]] | None = None, backend: Backend | None = None, replace_only_if_better: bool = True, max_threads: int | None = None)

    Synthesis for `Linear Function` circuits (blocks of `CX` and `SWAP` gates). Currently up to 9 qubit blocks.

    :param coupling_map: A list of pairs that represents physical links between qubits.
    :type coupling_map: list[list[int]], optional
    :param backend: Backend used for doing the AI Linear Function synthesis
    :type backend: Backend, optional
    :param replace_only_if_better: Determine if replace the original circuit with the synthesized one if it's better, defaults to True.
    :type replace_only_if_better: bool, optional
    :param max_threads: Set the number of requests to send in parallel.
    :type max_threads: int, optional
    """

    def __init__(
        self,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        backend: Backend | None = None,
        replace_only_if_better: bool = True,
        max_threads: int | None = None,
        local_mode: bool = True,
        **kwargs,
    ) -> None:
        ai_synthesis_provider = (
            AILocalLinearFunctionSynthesis()
            if local_mode
            else AILinearFunctionAPI(**kwargs)
        )

        super().__init__(
            synth_service=ai_synthesis_provider,
            coupling_map=coupling_map,
            backend=backend,
            replace_only_if_better=replace_only_if_better,
            max_threads=max_threads,
            local_mode=local_mode,
        )

    def _get_synth_input_and_original(self, node):
        return node.op, node.op.params[1]

    # Don't change this method name because it's replacing the method with the same name on the Qiskit Pass Manager
    def _get_nodes(self, dag):
        return dag.named_nodes("linear_function", "Linear_function")

    def _is_original_a_better_circuit(self, synth, original):
        return (
            original.decompose("swap").num_nonlocal_gates()
            <= synth.num_nonlocal_gates()
        )


class AIPermutationSynthesis(AISynthesis):
    """AIPermutationSynthesis(coupling_map: list[list[int]] | None = None, backend: Backend | None = None, replace_only_if_better: bool = True, max_threads: int | None = None)

    Synthesis for `Permutation` circuits (blocks of `SWAP` gates). Currently available for 65, 33, and 27 qubit blocks.

    :param coupling_map: A list of pairs that represents physical links between qubits.
    :type coupling_map: list[list[int]], optional
    :param backend: Backend used for doing the AI Permutation synthesis.
    :type backend: Backend, optional
    :param replace_only_if_better: Determine if replace the original circuit with the synthesized one if it's better, defaults to True.
    :type replace_only_if_better: bool, optional
    :param max_threads: Set the number of requests to send in parallel.
    :type max_threads: int, optional
    """

    def __init__(
        self,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        backend: Backend | None = None,
        replace_only_if_better: bool = True,
        max_threads: int | None = None,
        local_mode: bool = True,
        **kwargs,
    ) -> None:
        ai_synthesis_provider = (
            AILocalPermutationSynthesis() if local_mode else AIPermutationAPI(**kwargs)
        )

        super().__init__(
            synth_service=ai_synthesis_provider,
            coupling_map=coupling_map,
            backend=backend,
            replace_only_if_better=replace_only_if_better,
            max_threads=max_threads,
            local_mode=local_mode,
        )

    # Adding the original circuit creation
    def _get_synth_input_and_original(self, node):
        permutation = PermutationGate(node.op.params[0].tolist())
        len_perm = len(node.op.params[0].tolist())
        qubits = [i for i in range(0, len_perm)]
        circuit_orig = QuantumCircuit(len_perm)
        circuit_orig.append(permutation, qubits)
        circuit_orig = circuit_orig.decompose("permutation")
        return node.op.params[0].tolist(), circuit_orig

    def _get_nodes(self, dag):
        return dag.named_nodes("permutation", "Permutation")

    def _is_original_a_better_circuit(self, synth, original):
        return original.num_nonlocal_gates() <= synth.num_nonlocal_gates()


class AIPauliNetworkSynthesis(AISynthesis):
    """AIPauliNetworkSynthesis(coupling_map: list[list[int]] | None = None, backend: Backend | None = None, replace_only_if_better: bool = True, max_threads: int | None = None)

    Synthesis for `Pauli Networks` circuits (blocks of `H`, `S`, `SX`, `CX`, `RX`, `RY` and `RZ` gates). Currently up to 6 qubit blocks.

    :param coupling_map: A list of pairs that represents physical links between qubits.
    :type coupling_map: list[list[int]], optional
    :param backend: Backend used for doing the AI Pauli Network synthesis.
    :type backend: Backend, optional
    :param replace_only_if_better: Determine if replace the original circuit with the synthesized one if it's better, defaults to True.
    :type replace_only_if_better: bool, optional
    :param max_threads: Set the number of requests to send in parallel.
    :type max_threads: int, optional
    """

    def __init__(
        self,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        backend: Backend | None = None,
        replace_only_if_better: bool = True,
        max_threads: int | None = None,
        local_mode: bool = True,
        **kwargs,
    ) -> None:
        ai_synthesis_provider = (
            AILocalPauliNetworkSynthesis()
            if local_mode
            else AIPauliNetworkAPI(**kwargs)
        )

        super().__init__(
            synth_service=ai_synthesis_provider,
            coupling_map=coupling_map,
            backend=backend,
            replace_only_if_better=replace_only_if_better,
            max_threads=max_threads,
            local_mode=local_mode,
        )

    def _get_synth_input_and_original(self, node):
        input_circuit = node.op.params[1]
        return input_circuit, input_circuit

    def _get_nodes(self, dag):
        return dag.named_nodes("paulinetwork", "PauliNetwork")

    def _is_original_a_better_circuit(self, synth, original):
        # Select the best circuit
        score_synth = (
            (synth.num_nonlocal_gates(), synth.depth(lambda op: len(op.qubits) > 1))
            if synth is not None
            else (1e9, 1e9)
        )
        score_original = (
            (
                original.num_nonlocal_gates(),
                original.depth(lambda op: len(op.qubits) > 1),
            )
            if original is not None
            else (1e9, 1e9)
        )

        return score_synth > score_original
