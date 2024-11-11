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
from typing import Dict, List, Union, Literal

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, qasm2, qasm3
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit, library
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.qasm2 import QASM2ExportError, QASM2ParseError
from qiskit.transpiler import TranspileLayout
from qiskit.transpiler.layout import Layout

from qiskit_ibm_transpiler.wrappers import QiskitTranspilerService

# setting backoff logger to error level to avoid too much logging
logging.getLogger("backoff").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# TODO: Reuse this code, it's repeated several times
OptimizationOptions = Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]


class TranspileAPI(QiskitTranspilerService):
    """A helper class that covers some basic funcionality from the Qiskit Transpiler API"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transpile(
        self,
        circuits: Union[
            Union[List[str], str], Union[List[QuantumCircuit], QuantumCircuit]
        ],
        optimization_level: int = 1,
        optimization_preferences: Union[
            OptimizationOptions, List[OptimizationOptions], None
        ] = None,
        backend: Union[str, None] = None,
        coupling_map: Union[List[List[int]], None] = None,
        ai: Literal["true", "false", "auto"] = "true",
        qiskit_transpile_options: Dict = None,
        ai_layout_mode: str = None,
    ):
        circuits = circuits if isinstance(circuits, list) else [circuits]

        qasm_circuits = [_input_to_qasm(circ) for circ in circuits]

        body_params = {
            "qasm_circuits": qasm_circuits,
            "optimization_preferences": optimization_preferences,
        }

        if qiskit_transpile_options is not None:
            body_params["qiskit_transpile_options"] = qiskit_transpile_options
        if coupling_map is not None:
            body_params["backend_coupling_map"] = coupling_map

        query_params = {
            "backend": backend,
            "optimization_level": optimization_level,
            "ai": ai,
        }

        if ai_layout_mode is not None:
            query_params["ai_layout_mode"] = ai_layout_mode

        transpile_resp = self.request_and_wait(
            endpoint="transpile", body=body_params, params=query_params
        )

        logger.debug(f"transpile_resp={transpile_resp}")

        transpiled_circuits = []

        for res, orig_circ in zip(transpile_resp, circuits):
            try:
                transpiled_circuits.append(_get_circuit_from_result(res, orig_circ))
            except Exception as ex:
                logger.error("Error transforming the result to a QuantumCircuit object")
                raise

        return (
            transpiled_circuits
            if len(transpiled_circuits) > 1
            else transpiled_circuits[0]
        )

    def benchmark(
        self,
        circuits: Union[
            Union[List[str], str], Union[List[QuantumCircuit], QuantumCircuit]
        ],
        backend: str,
        optimization_level: int = 1,
        qiskit_transpile_options: Dict = None,
    ):
        raise Exception("Not implemented")


def _input_to_qasm(input_circ: Union[QuantumCircuit, str]):
    if isinstance(input_circ, QuantumCircuit):
        try:
            qasm = qasm2.dumps(input_circ).replace("\n", " ")
        except QASM2ExportError:
            qasm = qasm3.dumps(input_circ).replace("\n", " ")
    elif isinstance(input_circ, str):
        qasm = input_circ.replace("\n", " ")
    else:
        raise TypeError("Input circuits must be QuantumCircuit or qasm string.")
    return qasm


def _get_circuit_from_result(transpile_resp, orig_circuit):
    transpiled_circuit = _get_circuit_from_qasm(transpile_resp["qasm"])

    init_layout = transpile_resp["layout"]["initial"]
    final_layout = transpile_resp["layout"]["final"]

    orig_circuit = (
        _get_circuit_from_qasm(orig_circuit)
        if isinstance(orig_circuit, str)
        else orig_circuit
    )

    transpiled_circuit = QuantumCircuit(len(init_layout)).compose(transpiled_circuit)
    transpiled_circuit._layout = _create_transpile_layout(
        init_layout, final_layout, transpiled_circuit, orig_circuit
    )
    return transpiled_circuit


def _create_initial_layout(initial, n_used_qubits):
    """Create initial layout using the initial index layout and the number of active qubits."""
    total_qubits = len(initial)
    q_total = n_used_qubits
    a_total = total_qubits - q_total
    initial_layout = Layout()
    for q in range(q_total):
        initial_layout.add(initial[q], Qubit(QuantumRegister(q_total, "q"), q))
    for a in range(q_total, total_qubits):
        initial_layout.add(
            initial[a], Qubit(QuantumRegister(a_total, "ancilla"), a - q_total)
        )
    return initial_layout


def _create_input_qubit_mapping(qubits_used, total_qubits):
    """Create input qubit mapping with the number of active qubits and the total number of qubits."""
    input_qubit_mapping = {
        Qubit(QuantumRegister(qubits_used, "q"), q): q for q in range(qubits_used)
    }
    input_ancilla_mapping = {
        Qubit(
            QuantumRegister(total_qubits - qubits_used, "ancilla"), q - qubits_used
        ): q
        for q in range(qubits_used, total_qubits)
    }
    input_qubit_mapping.update(input_ancilla_mapping)
    return input_qubit_mapping


def _create_final_layout(initial, final, circuit):
    """Create final layout with the initial and final index layout and the circuit."""
    final_layout = Layout()
    q_total = len(initial)
    q_reg = QuantumRegister(q_total, "q")
    for i, j in zip(final, initial):
        q_index = circuit.find_bit(Qubit(q_reg, j)).index
        qubit = circuit.qubits[q_index]
        final_layout.add(i, qubit)

    return final_layout


def _create_transpile_layout(initial, final, circuit, orig_circuit):
    """Build the full transpile layout."""
    n_used_qubits = orig_circuit.num_qubits
    return TranspileLayout(
        initial_layout=_create_initial_layout(
            initial=initial, n_used_qubits=n_used_qubits
        ),  # final=final),
        input_qubit_mapping=_create_input_qubit_mapping(
            qubits_used=n_used_qubits, total_qubits=len(initial)
        ),
        final_layout=_create_final_layout(
            initial=initial, final=final, circuit=circuit
        ),
        _input_qubit_count=n_used_qubits,
        _output_qubit_list=circuit.qubits,
    )


class FixECR(TransformationPass):
    def run(self, dag):
        for node in dag.op_nodes():
            if node.name.startswith("ecr"):
                dag.substitute_node(node, library.ECRGate())
        return dag


def _get_circuit_from_qasm(qasm_string: str):
    try:
        return _get_circuit_from_qasm.fix_ecr(
            qasm2.loads(
                qasm_string,
                custom_instructions=_get_circuit_from_qasm.QISKIT_INSTRUCTIONS,
            )
        )
    except QASM2ParseError:
        return _get_circuit_from_qasm.fix_ecr(qasm3.loads(qasm_string))


_get_circuit_from_qasm.QISKIT_INSTRUCTIONS = list(qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
_get_circuit_from_qasm.QISKIT_INSTRUCTIONS.append(
    qasm2.CustomInstruction("ecr", 0, 2, library.ECRGate)
)
_get_circuit_from_qasm.fix_ecr = FixECR()
