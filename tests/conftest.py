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
import pytest
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import random_clifford
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService


@pytest.fixture(autouse=True)
def env_set(monkeypatch, request):
    if not "disable_monkeypatch" in request.keywords:
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_PERMUTATIONS_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/permutations",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_LINEAR_FUNCTIONS_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/linear_functions",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_CLIFFORD_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/clifford",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_ROUTING_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/routing",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_PAULI_NETWORK_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/pauli_network",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/",
        )
    logging.getLogger("qiskit_ibm_transpiler.ai.synthesis").setLevel(logging.DEBUG)


@pytest.fixture(scope="module")
def basic_cnot_circuit():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    return circuit


@pytest.fixture(scope="module")
def brisbane_backend_name():
    return "ibm_brisbane"


@pytest.fixture(scope="module")
def brisbane_backend(brisbane_backend_name):
    backend = QiskitRuntimeService().backend(brisbane_backend_name)

    return backend


@pytest.fixture(scope="module")
def brisbane_coupling_map(brisbane_backend):
    return brisbane_backend.coupling_map


@pytest.fixture(scope="module")
def brisbane_coupling_map_list_format(brisbane_backend):
    return list(brisbane_backend.coupling_map.get_edges())


@pytest.fixture(scope="module")
def permutation_circuit_brisbane(brisbane_backend):
    orig_qc = QuantumCircuit(brisbane_backend.num_qubits)
    # Add 8qL permutation to find subgraph in current models
    for i, p in enumerate([6, 2, 3, 4, 0, 1, 7, 5]):
        orig_qc.swap(i, p)
    for i, p in enumerate([7, 3, 4, 6, 0, 1, 2, 5]):
        starting_qubit = 37
        orig_qc.swap(i + starting_qubit, p + starting_qubit)
    for i, p in enumerate([5, 0, 4, 2, 6, 3, 1]):
        starting_qubit = 75
        orig_qc.swap(i + starting_qubit, p + starting_qubit)
    return orig_qc


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
