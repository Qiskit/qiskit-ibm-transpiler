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

"""Unit-testing linear_function_collection"""
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import CollectPermutations
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def test_permutation_collection_pass(random_circuit_transpiled):
    collect = PassManager(
        [
            CollectPermutations(),
        ]
    )
    collected_circuit = collect.run(random_circuit_transpiled)

    assert isinstance(collected_circuit, QuantumCircuit)


def test_permutation_collection_pass_collect(swap_circ):
    collect = PassManager(
        [
            CollectPermutations(min_block_size=1, max_block_size=5),
        ]
    )
    collected_circuit = collect.run(swap_circ)

    assert isinstance(collected_circuit, QuantumCircuit)
    assert any(g.operation.name.lower() == "permutation" for g in collected_circuit)


def test_permutation_collection_pass_no_collect(rzz_circ):
    collect = PassManager(
        [
            CollectPermutations(min_block_size=7, max_block_size=12),
        ]
    )
    collected_circuit = collect.run(rzz_circ)

    assert all(g.operation.name.lower() != "permutation" for g in collected_circuit)


def test_permutation_collection_max_block_size(swap_circ):
    collect = PassManager(
        [
            CollectPermutations(max_block_size=7),
        ]
    )
    collected_circuit = collect.run(swap_circ)

    assert all(len(g.qubits) <= 7 for g in collected_circuit)


def test_permutation_collection_min_block_size(swap_circ):
    collect = PassManager(
        [
            CollectPermutations(min_block_size=7, max_block_size=12),
        ]
    )
    collected_circuit = collect.run(swap_circ)

    assert all(
        len(g.qubits) >= 7 or g.operation.name.lower() != "permutation"
        for g in collected_circuit
    )


def test_permutation_collector(permutation_circuit, backend_27q, cmap_backend):
    qiskit_lvl3_transpiler = generate_preset_pass_manager(
        optimization_level=1, coupling_map=cmap_backend[backend_27q]
    )
    permutation_circuit = qiskit_lvl3_transpiler.run(permutation_circuit)

    pm = PassManager(
        [
            CollectPermutations(max_block_size=27),
        ]
    )
    perm_only_circ = pm.run(permutation_circuit)
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(perm_only_circ)
    perm_nodes = dag.named_nodes("permutation", "Permutation")
    assert len(perm_nodes) == 2
    assert perm_nodes[0].op.num_qubits == 27
    assert perm_nodes[1].op.num_qubits == 4
    assert not dag.named_nodes("linear_function", "Linear_function")
    assert not dag.named_nodes("clifford", "Clifford")
