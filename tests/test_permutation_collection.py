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

from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_transpiler.ai.collection import CollectPermutations


# TODO: Waiting for clarifications on what this test do
@pytest.mark.skip(reason="Commented asserts are not constant")
def test_permutation_collector(permutation_circuit_brisbane, brisbane_coupling_map):
    qiskit_lvl3_transpiler = generate_preset_pass_manager(
        optimization_level=1, coupling_map=brisbane_coupling_map
    )
    permutation_circuit = qiskit_lvl3_transpiler.run(permutation_circuit_brisbane)

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
