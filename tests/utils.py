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

from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford


def create_random_circuit(total_n_ubits, cliffords_n_qubits, clifford_num):
    circuit = QuantumCircuit(total_n_ubits)
    nq = cliffords_n_qubits
    for c in range(clifford_num):
        qs = np.random.choice(range(circuit.num_qubits), size=nq, replace=False)
        circuit.compose(
            random_clifford(nq, seed=42).to_circuit(), qubits=qs.tolist(), inplace=True
        )
        for q in qs:
            circuit.t(q)
    return circuit


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
