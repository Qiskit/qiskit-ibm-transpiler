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

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import random_clifford
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


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


@pytest.fixture(scope="module")
def random_circuit_transpiled(brisbane_coupling_map):
    circuit = create_random_circuit(27, 4, 2)
    qiskit_lvl3_transpiler = generate_preset_pass_manager(
        optimization_level=3, coupling_map=brisbane_coupling_map
    )
    return qiskit_lvl3_transpiler.run(circuit)


@pytest.fixture(scope="module")
def random_pauli_circuit_transpiled():
    from qiskit import qasm2

    circuit = qasm2.load(
        "tests/test_files/pauli_circuit.qasm",
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    return circuit


@pytest.fixture(scope="module")
def qv_circ():
    return QuantumVolume(10, depth=3, seed=42).decompose(reps=1)
