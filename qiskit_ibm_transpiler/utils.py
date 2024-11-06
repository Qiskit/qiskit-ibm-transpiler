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


"""
===============================================================================
Utilities (:mod:`qiskit_ibm_transpiler.utils`)
===============================================================================

.. currentmodule:: qiskit_ibm_transpiler.utils

Functions
=========

.. autofunction:: create_random_linear_function
.. autofunction:: get_metrics

"""

from typing import Dict, Union, List, Tuple

import numpy as np
import base64
import io
import struct
from qiskit import QuantumCircuit, qasm2, qasm3, qpy
from qiskit.circuit.library import LinearFunction
from qiskit.quantum_info import Clifford
from qiskit.synthesis.linear.linear_matrix_utils import random_invertible_binary_matrix
from qiskit.circuit import QuantumCircuit, library
from qiskit.transpiler.basepasses import TransformationPass


def get_metrics(qc: QuantumCircuit) -> Dict[str, int]:
    """Returns a dict with metrics from a QuantumCircuit"""
    qcd = qc.decompose(reps=3)
    return {
        "n_gates": qcd.size(),
        "n_layers": qcd.depth(),
        "n_cnots": qcd.num_nonlocal_gates(),
        "n_layers_cnots": qcd.depth(lambda x: x[0].name == "cx"),
    }


def random_permutation(n_qubits):
    """Generate a random permutation of n_qubits qubits."""
    return np.random.permutation(n_qubits)


def create_random_linear_function(n_qubits: int, seed: int = 123) -> LinearFunction:
    rand_lin = lambda seed: LinearFunction(
        random_invertible_binary_matrix(n_qubits, seed=seed)
    )

    return LinearFunction(rand_lin(seed))


def random_clifford_from_linear_function(n_qubits: int, seed: int = 123):
    """Generate a random clifford from a random linear function of n_qubits qubits."""

    random_linear = lambda seed: LinearFunction(
        random_invertible_binary_matrix(n_qubits, seed=seed)
    )
    random_clifford = Clifford(random_linear(seed))
    return random_clifford


def to_qasm3_iterative_decomposition(circuit: QuantumCircuit, n_iter: int = 10):
    decomposed_circuit = circuit.copy()
    for reps in range(1, n_iter + 1):
        qasm3_str = qasm3.dumps(decomposed_circuit)
        try:
            qasm3.loads(qasm3_str)
            break
        except qasm3.QASM3ImporterError:
            if reps == n_iter:
                raise qasm3.QASM3ExporterError(
                    f"Circuit couldn't be exported to QASM3, try using decompose() on your circuit"
                )
            decomposed_circuit = circuit.decompose(reps=reps)
    return qasm3_str.replace("\n", " ")


def input_to_qasm(input_circ: Union[QuantumCircuit, str]) -> str:
    if isinstance(input_circ, QuantumCircuit):
        try:
            qasm = qasm2.dumps(input_circ).replace("\n", " ")
        except qasm2.QASM2ExportError:
            qasm = to_qasm3_iterative_decomposition(input_circ)
    elif isinstance(input_circ, str):
        qasm = input_circ.replace("\n", " ")
    else:
        raise TypeError("Input circuits must be QuantumCircuit or qasm string.")
    return qasm


class FixECR(TransformationPass):
    def run(self, dag):
        for node in dag.op_nodes():
            if node.name.startswith("ecr"):
                dag.substitute_node(node, library.ECRGate())
        return dag


def get_circuit_from_qasm(qasm_string: str) -> QuantumCircuit:

    try:
        return get_circuit_from_qasm.fix_ecr(
            qasm2.loads(
                qasm_string,
                custom_instructions=get_circuit_from_qasm.QISKIT_INSTRUCTIONS,
            )
        )
    except qasm2.QASM2ParseError:
        return get_circuit_from_qasm.fix_ecr(qasm3.loads(qasm_string))


get_circuit_from_qasm.QISKIT_INSTRUCTIONS = list(qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
get_circuit_from_qasm.QISKIT_INSTRUCTIONS.append(
    qasm2.CustomInstruction("ecr", 0, 2, library.ECRGate)
)
get_circuit_from_qasm.fix_ecr = FixECR()


def get_qpy_from_circuit(
    input_circ: Union[QuantumCircuit, List[QuantumCircuit]]
) -> str:
    if isinstance(input_circ, QuantumCircuit) or isinstance(input_circ, list):
        output_b = io.BytesIO()
        qpy.dump(input_circ, output_b)
        qpy_string = base64.b64encode(output_b.getvalue()).decode("utf-8")
    else:
        raise TypeError(
            f"Input circuits must be QuantumCircuit or list of QuantumCircuits. Type provided: {type(input_circ)}"
        )
    return qpy_string


def get_circuit_from_qpy(qpy_string: str) -> QuantumCircuit:
    return qpy.load(io.BytesIO(base64.b64decode(qpy_string.encode("utf-8"))))[0]


def get_circuits_from_qpy(qpy_string: str) -> List[QuantumCircuit]:
    return qpy.load(io.BytesIO(base64.b64decode(qpy_string.encode("utf-8"))))


def serialize_circuit_to_qpy_or_qasm(
    input_circuit: QuantumCircuit,
) -> Tuple[Union[str, None], Union[str, None]]:
    qpy_result = None
    qasm_result = None
    try:
        qpy_result = get_qpy_from_circuit(input_circuit)
    except struct.error:
        qasm_result = input_to_qasm(input_circuit).replace("\n", " ")

    return qpy_result, qasm_result


def serialize_circuits_to_qpy_or_qasm(
    input_circuits: List[QuantumCircuit],
) -> Tuple[Union[str, None], Union[List[str], None]]:
    qpy_result = None
    qasm_result = None
    try:
        qpy_result = get_qpy_from_circuit(input_circuits)
    except struct.error:
        qasm_result = [
            input_to_qasm(the_circ).replace("\n", " ") for the_circ in input_circuits
        ]

    return qpy_result, qasm_result


def deserialize_circuit_from_qpy_or_qasm(
    qpy_input: Union[str, None], qasm_input: Union[str, None]
) -> Union[QuantumCircuit, None]:
    if qpy_input:
        return get_circuit_from_qpy(qpy_input)
    elif qasm_input:
        return get_circuit_from_qasm(qasm_input)
    else:
        return None
