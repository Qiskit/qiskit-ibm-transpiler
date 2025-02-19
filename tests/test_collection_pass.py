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

"""Tests for the collector pass"""

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_transpiler.ai.collection import CollectPermutations
from tests.parametrize_functions import (
    parametrize_circuit_collector_pass_and_operator_name,
    parametrize_collectable_gates_and_collector_pass,
    parametrize_collectable_gates_collector_pass_operation_name,
    parametrize_n_qubits,
    parametrize_non_collectable_gates_collector_pass_operation_name,
)
from tests.utils import create_linear_circuit


@parametrize_circuit_collector_pass_and_operator_name()
def test_collection_pass_collected_operators(
    circuit, collector_pass, operator_name, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_collector_pass = PassManager([collector_pass()])
    collected_circuit = custom_collector_pass.run(original_circuit)
    collected_circuit_instructions = collected_circuit.data
    operators_count = 0
    for circuit_instruction in collected_circuit_instructions:
        if operator_name in circuit_instruction.name:
            operators_count = operators_count + 1

    assert isinstance(collected_circuit, QuantumCircuit)
    assert operators_count == 2


@parametrize_collectable_gates_collector_pass_operation_name()
@parametrize_n_qubits()
def test_collection_pass_collectable_gates(
    collectable_gates, collector_pass, operation_name, n_qubits
):
    original_circuit = create_linear_circuit(n_qubits, collectable_gates)

    custom_collector_pass = PassManager(
        [
            collector_pass(min_block_size=1, max_block_size=5),
        ]
    )
    collected_circuit = custom_collector_pass.run(original_circuit)

    assert isinstance(collected_circuit, QuantumCircuit)
    assert any(g.operation.name.lower() == operation_name for g in collected_circuit)


@parametrize_non_collectable_gates_collector_pass_operation_name()
@parametrize_n_qubits()
def test_collection_pass_non_collectable_gates(
    non_collectable_gates, collector_pass, operation_name, n_qubits
):
    original_circuit = create_linear_circuit(n_qubits, non_collectable_gates)

    custom_collector_pass = PassManager(
        [
            collector_pass(min_block_size=7, max_block_size=12),
        ]
    )

    collected_circuit = custom_collector_pass.run(original_circuit)

    assert all(g.operation.name.lower() != operation_name for g in collected_circuit)


@parametrize_collectable_gates_and_collector_pass()
@parametrize_n_qubits()
def test_collection_pass_max_block_size(collectable_gates, collector_pass, n_qubits):
    original_circuit = create_linear_circuit(n_qubits, collectable_gates)

    custom_collector_pass = PassManager(
        [
            collector_pass(max_block_size=7),
        ]
    )
    collected_circuit = custom_collector_pass.run(original_circuit)

    assert all(len(g.qubits) <= 7 for g in collected_circuit)


@parametrize_collectable_gates_collector_pass_operation_name()
@parametrize_n_qubits()
def test_collection_pass_min_block_size(
    collectable_gates, collector_pass, operation_name, n_qubits
):
    original_circuit = create_linear_circuit(n_qubits, collectable_gates)

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


@parametrize_collectable_gates_and_collector_pass()
@parametrize_n_qubits()
@pytest.mark.timeout(10)
def test_collection_pass_with_barrier(collectable_gates, collector_pass, n_qubits):
    original_circuit = create_linear_circuit(n_qubits, collectable_gates)

    original_circuit.measure_all()
    collect = PassManager(
        [
            collector_pass(min_block_size=7, max_block_size=12),
        ]
    )
    # Without proper handling this test timeouts (actually the collect runs forever)
    collect.run(original_circuit)


# TODO: Waiting for clarifications on what this test do
@pytest.mark.skip(reason="Commented asserts are not constant")
def test_permutation_collector_pass(
    permutation_circuit_test_eagle, test_eagle_coupling_map
):
    qiskit_lvl3_transpiler = generate_preset_pass_manager(
        optimization_level=1, coupling_map=test_eagle_coupling_map
    )
    permutation_circuit = qiskit_lvl3_transpiler.run(permutation_circuit_test_eagle)

    pm = PassManager(
        [
            CollectPermutations(max_block_size=27),
        ]
    )
    perm_only_circ = pm.run(permutation_circuit)
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(perm_only_circ)
    perm_nodes = dag.named_nodes("permutation", "Permutation")
    assert len(perm_nodes) == 3
    # assert perm_nodes[0].op.num_qubits == 12
    # assert perm_nodes[1].op.num_qubits == 8
    assert not dag.named_nodes("linear_function", "Linear_function")
    assert not dag.named_nodes("clifford", "Clifford")
