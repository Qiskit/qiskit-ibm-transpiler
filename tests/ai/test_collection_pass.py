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

"""Unit-testing collection"""
import pytest

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import CollectPermutations
from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
from qiskit_ibm_transpiler.ai.collection import CollectCliffords
from qiskit_ibm_transpiler.ai.collection import CollectPauliNetworks


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


@pytest.mark.parametrize(
    "collector_pass",
    [
        (CollectPermutations),
        (CollectLinearFunctions),
        (CollectCliffords),
        (CollectPauliNetworks),
    ],
    ids=["permutation", "linear_function", "clifford", "pauli_network"],
)
def test_collection_pass(random_circuit_transpiled, collector_pass):
    original_circuit = random_circuit_transpiled

    custom_collector_pass = PassManager([collector_pass()])

    collected_circuit = custom_collector_pass.run(original_circuit)

    assert isinstance(collected_circuit, QuantumCircuit)


@pytest.mark.parametrize(
    "circuit_gates, collector_pass, operation_name",
    [
        ("swap", CollectPermutations, "permutation"),
        ("cx", CollectLinearFunctions, "linear_function"),
        ("cz", CollectCliffords, "clifford"),
        ("swap", CollectPauliNetworks, "paulinetwork"),
    ],
    ids=["permutation", "linear_function", "clifford", "pauli_network"],
)
@pytest.mark.parametrize("n_qubits", [3, 10, 30])
def test_collection_pass_collect(
    circuit_gates, collector_pass, operation_name, n_qubits
):
    original_circuit = create_linear_circuit(n_qubits, circuit_gates)

    custom_collector_pass = PassManager(
        [
            collector_pass(min_block_size=1, max_block_size=5),
        ]
    )
    collected_circuit = custom_collector_pass.run(original_circuit)

    assert isinstance(collected_circuit, QuantumCircuit)
    assert any(g.operation.name.lower() == operation_name for g in collected_circuit)


@pytest.mark.parametrize(
    "circuit_gates, collector_pass, operation_name",
    [
        ("rzz", CollectPermutations, "permutation"),
        ("rzz", CollectLinearFunctions, "linear_function"),
        ("rzz", CollectCliffords, "clifford"),
        ("t", CollectPauliNetworks, "paulinetwork"),
    ],
    ids=["permutation", "linear_function", "clifford", "pauli_network"],
)
@pytest.mark.parametrize("n_qubits", [3, 10, 30])
def test_collection_pass_no_collect(
    circuit_gates, collector_pass, operation_name, n_qubits
):
    original_circuit = create_linear_circuit(n_qubits, circuit_gates)

    custom_collector_pass = PassManager(
        [
            collector_pass(min_block_size=7, max_block_size=12),
        ]
    )

    collected_circuit = custom_collector_pass.run(original_circuit)

    assert all(g.operation.name.lower() != operation_name for g in collected_circuit)


@pytest.mark.parametrize(
    "circuit_gates, collector_pass",
    [
        ("swap", CollectPermutations),
        ("cx", CollectLinearFunctions),
        ("cz", CollectCliffords),
        ("swap", CollectPauliNetworks),
    ],
    ids=["permutation", "linear_function", "clifford", "pauli_network"],
)
@pytest.mark.parametrize("n_qubits", [3, 10, 30])
def test_collection_max_block_size(circuit_gates, collector_pass, n_qubits):
    original_circuit = create_linear_circuit(n_qubits, circuit_gates)

    custom_collector_pass = PassManager(
        [
            collector_pass(max_block_size=7),
        ]
    )
    collected_circuit = custom_collector_pass.run(original_circuit)

    assert all(len(g.qubits) <= 7 for g in collected_circuit)


@pytest.mark.parametrize(
    "circuit_gates, collector_pass, operation_name",
    [
        ("swap", CollectPermutations, "permutation"),
        ("cx", CollectLinearFunctions, "linear_function"),
        ("cz", CollectCliffords, "clifford"),
        ("swap", CollectPauliNetworks, "paulinetwork"),
    ],
    ids=["permutation", "linear_function", "clifford", "pauli_network"],
)
@pytest.mark.parametrize("n_qubits", [3, 10, 30])
def test_collection_min_block_size(
    circuit_gates, collector_pass, operation_name, n_qubits
):
    original_circuit = create_linear_circuit(n_qubits, circuit_gates)

    custom_collector_pass = PassManager(
        [
            collector_pass(min_block_size=7, max_block_size=12),
        ]
    )
    collected_circuit = custom_collector_pass.run(original_circuit)

    assert all(
        len(g.qubits) >= 7 or g.operation.name.lower() != operation_name
        for g in collected_circuit
    )


@pytest.mark.parametrize(
    "circuit_gates, collector_pass",
    [
        ("swap", CollectPermutations),
        ("cx", CollectLinearFunctions),
        ("cz", CollectCliffords),
        ("swap", CollectPauliNetworks),
    ],
    ids=["permutation", "linear_function", "clifford", "pauli_network"],
)
@pytest.mark.parametrize("n_qubits", [3, 10, 30])
@pytest.mark.timeout(10)
def test_collection_with_barrier(circuit_gates, collector_pass, n_qubits):
    original_circuit = create_linear_circuit(n_qubits, circuit_gates)

    original_circuit.measure_all()
    collect = PassManager(
        [
            collector_pass(min_block_size=7, max_block_size=12),
        ]
    )
    # Without proper handling this test timeouts (actually the collect runs forever)
    collect.run(original_circuit)
