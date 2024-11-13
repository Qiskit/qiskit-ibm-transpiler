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
import random
from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import random_clifford
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit.circuit.library.standard_gates import (
    CXGate,
    HGate,
    IGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SXGate,
    XGate,
    YGate,
    ZGate,
)

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

base_gate_dict = {
    "h": HGate,
    "s": SGate,
    "z": ZGate,
    "x": XGate,
    "y": YGate,
    "z": YGate,
    "sx": SXGate,
    "cx": CXGate,
    "i": IGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
}
allowed_gates = ["cx", "x", "y", "z", "s", "sx", "h"]
allowed_rots = ["rx", "ry", "rz"]


def apply_random_gate(num_qubits, qc):
    gate = np.random.choice(allowed_gates)
    match gate:
        case "cx":
            qubits = tuple(np.random.choice(range(num_qubits), size=2, replace=False))
            qc.cx(*qubits)
        case "x" | "y" | "z" | "s" | "sx" | "h":
            qubit = (np.random.choice(range(num_qubits)),)
            qc.append(base_gate_dict[gate](), qubit)


def apply_random_rot(num_qubits, qc):
    gate = np.random.choice(allowed_rots)
    match gate:
        case "rx" | "ry" | "rz":
            qubit = (np.random.choice(range(num_qubits)),)
            angle = np.random.uniform(-np.pi, np.pi)
            qc.append(base_gate_dict[gate](angle), qubit)


def get_random_pauli_network(num_qubits, depth=3, rot_p=0.4, max_rots=10, seed=42):
    np.random.seed(seed)
    qc = QuantumCircuit(num_qubits)
    rotations = 0
    while qc.depth() < depth and rotations <= max_rots:
        if np.random.uniform(0, 1) > rot_p:
            apply_random_gate(num_qubits, qc)
        else:
            if rotations == max_rots:
                continue
            apply_random_rot(num_qubits, qc)
            rotations += 1
    return qc

@pytest.fixture(scope="module")
def random_pauli_circuit_transpiled(backend_27q, cmap_backend):
    circuit = get_random_pauli_network(27,30)
    qiskit_lvl3_transpiler = generate_preset_pass_manager(
        optimization_level=1, coupling_map=cmap_backend[backend_27q]
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
