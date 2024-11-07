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


@pytest.fixture(scope="module")
def random_circuit_transpiled(backend_27q, cmap_backend):
    circuit = create_random_circuit(27, 4, 2)
    qiskit_lvl3_transpiler = generate_preset_pass_manager(
        optimization_level=3, coupling_map=cmap_backend[backend_27q]
    )
    return qiskit_lvl3_transpiler.run(circuit)


@pytest.fixture(scope="module")
def qv_circ():
    return QuantumVolume(10, depth=3, seed=42).decompose(reps=1)


@pytest.fixture(scope="module", params=[3, 10, 30])
def cnot_circ(request):
    return create_linear_circuit(request.param, "cx")


@pytest.fixture(scope="module", params=[3, 10, 30])
def swap_circ(request):
    return create_linear_circuit(request.param, "swap")


@pytest.fixture(scope="module", params=[3, 10, 30])
def cz_circ(request):
    return create_linear_circuit(request.param, "cz")


@pytest.fixture(scope="module", params=[3, 10, 30])
def rzz_circ(request):
    return create_linear_circuit(request.param, "rzz")

@pytest.fixture(scope="module", params=[3, 10, 30])
def t_circ(request):
    return create_linear_circuit(request.param, "t")
