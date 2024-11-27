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

"""Functions used on the tests"""
import numpy as np
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit.library import Permutation, LinearFunction
from qiskit.quantum_info import random_clifford, random_pauli


def create_random_circuit_with_several_operators(
    operator: str,
    n_qubits_circuit: int,
    n_qubits_operator: int,
    n_operator_circuits: int,
):
    circuit = QuantumCircuit(n_qubits_circuit)
    np.random.seed(42)
    for c in range(n_operator_circuits):
        qs = np.random.choice(
            range(n_qubits_circuit), size=n_qubits_operator, replace=False
        )

        operator_circuit = create_operator_circuit(operator, n_qubits_operator)
        circuit.compose(
            operator_circuit,
            qubits=qs.tolist(),
            inplace=True,
        )
        for q in qs:
            circuit.t(q)

    return circuit


def create_operator_circuit(operator: str, num_qubits: int):
    if operator == "Permutation":
        return Permutation(num_qubits, seed=42)
    elif operator == "LinearFunction":
        np.random.seed(42)
        matrix = np.random.randint(2, size=(num_qubits, num_qubits))
        circuit = QuantumCircuit(num_qubits)
        circuit.append(LinearFunction(matrix), range(num_qubits))
        return circuit
    elif operator == "Clifford":
        return random_clifford(num_qubits, seed=42).to_circuit()
    elif operator == "PauliNetwork":
        return random_pauli(num_qubits, seed=42)
    else:
        raise ValueError(f"Unsopported operator {operator}")


def create_linear_circuit(n_qubits, gates):
    circuit = QuantumCircuit(n_qubits)
    for q in range(n_qubits - 1):
        if gates == "cx":
            circuit.cx(q, q + 1)
        elif gates == "swap":
            circuit.swap(q, q + 1)
        elif gates == "cz":
            circuit.cz(q, q + 1)
        elif gates == "rzz":
            circuit.rzz(1.23, q, q + 1)
        elif gates == "t":
            circuit.t(q)
    return circuit
