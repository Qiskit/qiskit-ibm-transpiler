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
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import CollectPauliNetworks


def test_pauli_network_collection_pass(random_circuit_transpiled):
    collect = PassManager(
        [
            CollectPauliNetworks(),
        ]
    )
    collected_circuit = collect.run(random_circuit_transpiled)

    assert isinstance(collected_circuit, QuantumCircuit)


def test_pauli_network_collection_pass_collect(swap_circ):
    collect = PassManager(
        [
            CollectPauliNetworks(min_block_size=1, max_block_size=5),
        ]
    )
    collected_circuit = collect.run(swap_circ)

    assert isinstance(collected_circuit, QuantumCircuit)
    assert any(g.operation.name.lower() == "paulinetwork" for g in collected_circuit)


def test_pauli_network_collection_pass_no_collect(t_circ):
    collect = PassManager(
        [
            CollectPauliNetworks(min_block_size=7, max_block_size=12),
        ]
    )
    collected_circuit = collect.run(t_circ)

    assert all(g.operation.name.lower() != "paulinetwork" for g in collected_circuit)


def test_pauli_network_collection_max_block_size(swap_circ):
    collect = PassManager(
        [
            CollectPauliNetworks(max_block_size=7),
        ]
    )
    collected_circuit = collect.run(swap_circ)

    assert all(len(g.qubits) <= 7 for g in collected_circuit)


def test_pauli_network_collection_min_block_size(swap_circ):
    collect = PassManager(
        [
            CollectPauliNetworks(min_block_size=7, max_block_size=12),
        ]
    )
    collected_circuit = collect.run(swap_circ)

    assert all(
        len(g.qubits) >= 7 or g.operation.name.lower() != "paulinetwork"
        for g in collected_circuit
    )
