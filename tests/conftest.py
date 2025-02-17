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

"""Fixtures used on the tests"""

import logging

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit_ibm_transpiler.utils import (
    create_random_linear_function,
    get_qpy_from_circuit,
    random_clifford_from_linear_function,
)
from tests.utils import create_random_circuit_with_several_operators


@pytest.fixture(autouse=True)
def env_set(monkeypatch, request):
    if "disable_monkeypatch" not in request.keywords:
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_PERMUTATIONS_URL",
            "https://cloud-transpiler.quantum-computing.ibm.com/permutations",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_LINEAR_FUNCTIONS_URL",
            "https://cloud-transpiler.quantum-computing.ibm.com/linear_functions",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_CLIFFORD_URL",
            "https://cloud-transpiler.quantum-computing.ibm.com/clifford",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_ROUTING_URL",
            "https://cloud-transpiler.quantum-computing.ibm.com/routing",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_PAULI_NETWORK_URL",
            "https://cloud-transpiler.quantum-computing.ibm.com/pauli_network",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_URL",
            "https://cloud-transpiler.quantum-computing.ibm.com/",
        )
    logging.getLogger("qiskit_ibm_transpiler.ai.synthesis").setLevel(logging.DEBUG)


@pytest.fixture(scope="module")
def non_valid_backend_name():
    return "ibm_torin"


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
def brisbane_num_qubits(brisbane_backend):
    return brisbane_backend.num_qubits


@pytest.fixture(scope="module")
def qpy_circuit_with_transpiling_error():
    return "UUlTS0lUDAECAAAAAAAAAAABZXEAC2YACAAAAAMAAAAAAAAAAAAAAAIAAAABAAAAAAAAAAYAAAAAY2lyY3VpdC0xNjAAAAAAAAAAAHt9cQEAAAADAAEBcQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAABACRnAAAAAgAAAAABAAAAAAAAALsAAAAAAAAAAAAAAAAAAAAAZGN4X2ZkN2JlNTcxOTQ1OTRkZTY5ZDFlMTNmNmFmYmY1NmZjAAtmAAgAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAGNpcmN1aXQtMTYxAAAAAAAAAAB7fQAAAAAAAAAAAAYAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAEAAAABQ1hHYXRlcQAAAABxAAAAAQAGAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAABAAAAAUNYR2F0ZXEAAAABcQAAAAAAAAD///////////////8AAAAAAAAAAAAGAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAABAAAAAUNaR2F0ZXEAAAAAcQAAAAIABwAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABTZGdHYXRlcQAAAAEAJAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABkY3hfZmQ3YmU1NzE5NDU5NGRlNjlkMWUxM2Y2YWZiZjU2ZmNxAAAAAnEAAAABAAYAAAADAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVTNHYXRlcQAAAABmAAAAAAAAAAiMHTg9AR8PQGYAAAAAAAAACH+xa3jilAtAZgAAAAAAAAAItmSFHpiI8j8ABwAAAAEAAAACAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAFDUlhHYXRlcQAAAAFxAAAAAGYAAAAAAAAACI2Sd1JN3gJAAAUAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWUdhdGVxAAAAAgAAAP///////////////wAAAAAAAAAA"


@pytest.fixture(scope="module")
def non_valid_qpy_circuit(basic_cnot_circuit):
    return [get_qpy_from_circuit(basic_cnot_circuit)] * 2


@pytest.fixture(scope="module")
def permutation_circuit_brisbane(brisbane_num_qubits):
    orig_qc = QuantumCircuit(brisbane_num_qubits)
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


# TODO: All the tests that use this circuit keeps the original circuit. Check if this is the better option
# for doing those tests
@pytest.fixture(scope="module")
def random_linear_function_circuit():
    circuit = QuantumCircuit(8)
    linear_function = create_random_linear_function(8)
    circuit.append(linear_function, range(8))
    # Using decompose since we need a QuantumCircuit, not a LinearFunction. We created an empty
    # circuit, so it contains only a LinearFunction
    circuit = circuit.decompose(reps=1)

    return circuit


# TODO: All the tests that use this circuit keeps the original circuit. Check if this is the better option
# for doing those tests
@pytest.fixture(scope="module")
def random_clifford_circuit():
    circuit = QuantumCircuit(8)
    clifford = random_clifford_from_linear_function(8)
    circuit.append(clifford, range(8))
    # Using decompose since we need a QuantumCircuit, not a Clifford. We created an empty
    # circuit, so it contains only a Clifford
    circuit = circuit.decompose(reps=1)

    return circuit


@pytest.fixture(scope="module")
def random_brisbane_circuit_with_two_cliffords(brisbane_num_qubits):
    return create_random_circuit_with_several_operators(
        "Clifford", brisbane_num_qubits, 4, 2
    )


@pytest.fixture(scope="module")
def random_brisbane_circuit_with_two_linear_functions(brisbane_num_qubits):
    return create_random_circuit_with_several_operators(
        "LinearFunction", brisbane_num_qubits, 4, 2
    )


@pytest.fixture(scope="module")
def random_brisbane_circuit_with_two_permutations(brisbane_num_qubits):
    return create_random_circuit_with_several_operators(
        "Permutation", brisbane_num_qubits, 4, 2
    )


@pytest.fixture(scope="module")
def random_brisbane_circuit_with_two_paulis(brisbane_num_qubits):
    return create_random_circuit_with_several_operators(
        "PauliNetwork", brisbane_num_qubits, 4, 2
    )


@pytest.fixture(scope="module")
def pauli_circuit_transpiled():
    from qiskit import qasm2

    circuit = qasm2.load(
        "tests/test_files/pauli_circuit.qasm",
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )

    return circuit


@pytest.fixture(scope="module")
def qv_circ():
    return QuantumVolume(10, depth=3, seed=42).decompose(reps=1)


@pytest.fixture(scope="module")
def basic_cnot_circuit():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    return circuit


@pytest.fixture(scope="module")
def basic_swap_circuit():
    circuit = QuantumCircuit(3)
    circuit.swap(0, 1)
    circuit.swap(1, 2)

    return circuit


@pytest.fixture(scope="module")
def circuit_with_barrier():
    circuit = QuantumCircuit(5)
    circuit.x(4)
    circuit.barrier()
    circuit.z(3)
    circuit.cx(3, 4)

    return circuit
