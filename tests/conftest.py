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
from qiskit import QuantumCircuit
from qiskit.transpiler.coupling import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeQuebec
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
def backend():
    return "ibm_quebec"


@pytest.fixture(scope="module")
def backend_27q():
    return "ibm_peekskill"


@pytest.fixture(scope="module")
def peekskill_coupling_map_list_format(backend_27q, cmap_backend):
    coupling_map = cmap_backend[backend_27q]

    return list(coupling_map.get_edges())


@pytest.fixture(scope="module")
def coupling_map():
    return FakeQuebec().coupling_map


@pytest.fixture(scope="module")
def cmap_backend():
    return {
        "ibm_peekskill": CouplingMap(
            [
                [0, 1],
                [1, 0],
                [1, 2],
                [1, 4],
                [2, 1],
                [2, 3],
                [3, 2],
                [3, 5],
                [4, 1],
                [4, 7],
                [5, 3],
                [5, 8],
                [6, 7],
                [7, 4],
                [7, 6],
                [7, 10],
                [8, 5],
                [8, 9],
                [8, 11],
                [9, 8],
                [10, 7],
                [10, 12],
                [11, 8],
                [11, 14],
                [12, 10],
                [12, 13],
                [12, 15],
                [13, 12],
                [13, 14],
                [14, 11],
                [14, 13],
                [14, 16],
                [15, 12],
                [15, 18],
                [16, 14],
                [16, 19],
                [17, 18],
                [18, 15],
                [18, 17],
                [18, 21],
                [19, 16],
                [19, 20],
                [19, 22],
                [20, 19],
                [21, 18],
                [21, 23],
                [22, 19],
                [22, 25],
                [23, 21],
                [23, 24],
                [24, 23],
                [24, 25],
                [25, 22],
                [25, 24],
                [25, 26],
                [26, 25],
            ]
        ),
        "ibm_quebec": CouplingMap(
            [
                [1, 0],
                [2, 1],
                [3, 2],
                [3, 4],
                [4, 15],
                [5, 4],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 16],
                [9, 8],
                [9, 10],
                [11, 10],
                [11, 12],
                [13, 12],
                [14, 0],
                [14, 18],
                [16, 26],
                [17, 12],
                [19, 18],
                [19, 20],
                [20, 33],
                [21, 20],
                [21, 22],
                [22, 15],
                [23, 22],
                [24, 23],
                [25, 24],
                [26, 25],
                [26, 27],
                [28, 27],
                [28, 35],
                [29, 28],
                [30, 17],
                [30, 29],
                [31, 30],
                [31, 32],
                [34, 24],
                [34, 43],
                [35, 47],
                [36, 32],
                [38, 37],
                [38, 39],
                [39, 33],
                [39, 40],
                [40, 41],
                [41, 42],
                [43, 42],
                [43, 44],
                [44, 45],
                [46, 45],
                [47, 46],
                [48, 47],
                [48, 49],
                [50, 49],
                [51, 36],
                [51, 50],
                [52, 37],
                [52, 56],
                [53, 41],
                [53, 60],
                [54, 45],
                [54, 64],
                [55, 49],
                [55, 68],
                [57, 56],
                [58, 57],
                [58, 59],
                [59, 60],
                [60, 61],
                [62, 61],
                [62, 63],
                [62, 72],
                [64, 63],
                [65, 64],
                [65, 66],
                [66, 67],
                [66, 73],
                [68, 67],
                [69, 68],
                [70, 69],
                [70, 74],
                [71, 58],
                [71, 77],
                [74, 89],
                [75, 76],
                [76, 77],
                [77, 78],
                [79, 78],
                [80, 79],
                [80, 81],
                [81, 72],
                [82, 81],
                [82, 83],
                [83, 84],
                [85, 73],
                [85, 84],
                [85, 86],
                [86, 87],
                [87, 88],
                [87, 93],
                [89, 88],
                [90, 75],
                [91, 79],
                [91, 98],
                [92, 83],
                [94, 90],
                [94, 95],
                [96, 95],
                [96, 109],
                [97, 96],
                [97, 98],
                [98, 99],
                [100, 99],
                [100, 110],
                [101, 100],
                [101, 102],
                [102, 92],
                [103, 102],
                [103, 104],
                [104, 105],
                [106, 93],
                [106, 105],
                [106, 107],
                [108, 107],
                [108, 112],
                [109, 114],
                [110, 118],
                [111, 104],
                [114, 113],
                [115, 114],
                [115, 116],
                [116, 117],
                [118, 117],
                [118, 119],
                [120, 119],
                [121, 120],
                [122, 111],
                [122, 121],
                [123, 122],
                [123, 124],
                [124, 125],
                [126, 112],
                [126, 125],
            ]
        ),
    }


@pytest.fixture
def basic_cnot_circuit():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    return circuit


@pytest.fixture
def brisbane_backend():
    backend = QiskitRuntimeService().backend("ibm_brisbane")

    return backend


# TODO: All the tests that use this circuit keeps the original circuit. Check if this is the better option
# for doing those tests
@pytest.fixture
def permutation_circuit(peekskill_coupling_map_list_format):
    circuit = QuantumCircuit(27)
    coupling_map = peekskill_coupling_map_list_format

    for i, j in coupling_map:
        circuit.h(i)
        circuit.cx(i, j)
    for i, j in coupling_map:
        circuit.swap(i, j)
    for i, j in coupling_map:
        circuit.h(i)
        circuit.cx(i, j)
    for i, j in coupling_map[:4]:
        circuit.swap(i, j)
    for i, j in coupling_map:
        circuit.cx(i, j)

    return circuit


@pytest.fixture
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
