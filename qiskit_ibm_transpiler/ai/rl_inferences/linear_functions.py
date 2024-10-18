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

"""Synthesize module"""
from ai_synthesis_py import LinFuncSynthesis
import logging

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library.generalized_gates.linear_function import LinearFunction
from qiskit.quantum_info import Clifford

from qiskit_ibm_transpiler.utils import (
    embed_clifford,
    check_synthesized_clifford,
    check_topology_synthesized_circuit,
)
from qiskit_ibm_transpiler.ai.models.linear_functions import (
    MODEL_CMAPS,
    MODEL_HASHES,
    MODEL_NAMES,
    MODEL_QUBITS,
)

logger = logging.getLogger(__name__)


def perm_linear(linear, perm):
    # perm_inv = np.argsort(perm)
    return linear[:, perm][perm, :]


def to_circuit(synth_out, num_qubits=None):
    if num_qubits is None:
        num_qubits = max(max(qs) for qs in synth_out) + 1
    circuit = QuantumCircuit(num_qubits)
    # circuit.barrier()
    for qs in synth_out:
        circuit.cx(*qs)
    # circuit.barrier()
    return circuit


def select_model_id(
    model_name: str = None, coupling_map_hash: str = None, topology: str = None
) -> int:
    selected_model_id = None

    if model_name in MODEL_NAMES:
        selected_model_id = MODEL_NAMES.index(model_name)
    elif coupling_map_hash in MODEL_HASHES:
        selected_model_id = MODEL_HASHES.index(coupling_map_hash)
    # Topology is for now the name of the model
    elif topology in MODEL_NAMES:
        selected_model_id = MODEL_NAMES.index(topology)

    return selected_model_id


def check_inference_result(
    rl_circuit: QuantumCircuit,
    cliff: Clifford,
    model_name: str = None,
    backend: str = None,
    topology: str = None,
    selected_model_id: int | None = None,
):
    logger.debug("Checking results from synthesis process")
    if not check_synthesized_clifford(cliff, rl_circuit):
        logger.warning(
            "The check for synthesized clifford circuit vs the original clifford failed"
        )
        return None

    if topology is not None or backend is not None:
        logger.debug(
            "Checking if synthesized clifford follows the input topology/backend's topology"
        )
        logger.debug(f"Circuit: {qasm2.dumps(rl_circuit)}")
        logger.debug(
            f"Topology: {model_name}. Coupling map: {MODEL_CMAPS[selected_model_id]}"
        )
        if not check_topology_synthesized_circuit(
            rl_circuit,
            MODEL_CMAPS[selected_model_id],
        ):
            logger.warning(
                "The check to evaluate if synthesized clifford circuit follows the topology failed"
            )
            return None
    logger.debug("Success. The checks for the synthesis results were successful")


class LinearFunctionInference:
    RL_INFERENCE = None
    DEFAULT_N_STEPS = 10
    MODEL_2Q = {
        (1, 0, 0, 1): [],
        (1, 0, 1, 1): [(0, 1)],
        (1, 1, 0, 1): [(1, 0)],
        (0, 1, 1, 1): [(0, 1), (1, 0)],
        (1, 1, 1, 0): [(1, 0), (0, 1)],
        (0, 1, 1, 0): [(0, 1), (1, 0), (0, 1)],
    }

    def __init__(self):
        if not LinearFunctionInference.RL_INFERENCE:
            LinearFunctionInference.RL_INFERENCE = LinFuncSynthesis()

    def synthesize(
        self,
        cliff: Clifford,
        model_name: str = None,
        coupling_map_hash: str = None,
        backend: str = None,
        topology: str = None,
        n_qubits: int = None,
        check_result: bool = False,
        n_steps: int | None = None,
    ) -> QuantumCircuit | None:
        """Synthesize Clifford using the RL model"""
        selected_model_id = select_model_id(
            model_name=model_name,
            coupling_map_hash=coupling_map_hash,
            topology=topology,
        )

        if selected_model_id is None:
            logger.warning(
                "The model selected for inference is not available. Options used: "
                f"model_name: {model_name}, backend: {backend}, topology: {topology}, "
                f"coupling_map_hash: {coupling_map_hash}, "
                f"n_qubits: {n_qubits}, circuit n_qubits: {cliff.num_qubits} "
            )
            return None

        model_name = MODEL_NAMES[selected_model_id]
        model_n_qubits = MODEL_QUBITS[selected_model_id]
        circ_n_qubits = cliff.num_qubits
        if model_n_qubits < circ_n_qubits:
            logger.warning(
                f"The model selected for inference '{model_name}' cannot synthesize that circuit. "
                f"The model is trained for n_qubits={model_n_qubits} while the "
                f"circuits uses n_qubits={circ_n_qubits} "
            )
            return None

        # We should embed clifford also in the cases of circuits with less qubits
        # than the bakend's or topologies qubits, or the model selected manually
        if model_n_qubits > circ_n_qubits:
            cliff = embed_clifford(cliff=cliff, nq=model_n_qubits)

        if n_steps is None:
            n_steps = self.DEFAULT_N_STEPS

        logger.info(
            f"Synthesizing the Clifford using the model {model_name}. "
            # f"Launching {n_steps} n_steps with n_envs {selected_model[0].info.n_envs}..."
        )
        logger.debug("Instantiating make_rl_shots_decomposer synthesis function")

        logger.debug("Synthesizing circuit")
        synth_input = LinearFunction(cliff).linear  # TODO: Get the right format for

        # Done in the outher function
        # synth_input_perm = perm_linear(synth_input, subgraph_perm)

        # Synth the input
        if selected_model_id == 29:  # Hack for the 2qL topology
            linear_flat = tuple(synth_input.astype(int).flatten().tolist())
            gate_list = self.MODEL_2Q[linear_flat]
        else:
            linear_flat = synth_input.astype(float).flatten().tolist()
            # Is 10 the number of steps?
            gate_list = list(
                reversed(self.RL_INFERENCE.run(selected_model_id, 10, linear_flat))
            )
        if len(gate_list) == 0:
            return QuantumCircuit(model_n_qubits)
        rl_circuit = to_circuit(gate_list, model_n_qubits)
        logger.debug("Circuit synthesized")

        if check_result:
            check_inference_result(
                rl_circuit,
                cliff=cliff,
                model_name=model_name,
                backend=backend,
                topology=topology,
                selected_model_id=selected_model_id,
            )

        return rl_circuit
