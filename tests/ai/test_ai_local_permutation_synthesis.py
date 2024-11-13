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

"""Testing AI local synthesis for linear functions"""
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import CollectPermutations
from qiskit_ibm_transpiler.ai.synthesis import AIPermutationSynthesis


def test_ai_local_permutation_synthesis_wrong_backend(basic_cnot_circuit):
    original_circuit = basic_cnot_circuit

    with pytest.raises(
        PermissionError,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        ai_permutations_synthesis_pass = PassManager(
            [
                CollectPermutations(min_block_size=2),
                AIPermutationSynthesis(backend_name="wrong_backend"),
            ]
        )

        ai_permutations_synthesis_pass.run(original_circuit)


@pytest.mark.skip(
    reason="The original circuit doesn't return a DAGCircuit with nodes. We are deciding how the code should behave on this case"
)
def test_ai_local_permutation_synthesis_returns_original_circuit(
    basic_cnot_circuit, caplog
):
    # When the original circuit is better than the synthesized one, we keep the original

    original_circuit = basic_cnot_circuit

    ai_permutations_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2),
            AIPermutationSynthesis(backend_name="ibm_brisbane"),
        ]
    )

    synthesized_circuit = ai_permutations_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
    assert synthesized_circuit == original_circuit
    assert "Keeping the original circuit" in caplog.text


@pytest.mark.skip(
    reason="The original circuit doesn't return a DAGCircuit with nodes. We are deciding how the code should behave on this case"
)
def test_ai_local_permutation_synthesis_dont_returns_original_circuit(
    basic_cnot_circuit, caplog
):
    # When the original circuit is better than the synthesized one,
    # but replace_only_if_better is False, we return the synthesized circuit

    original_circuit = basic_cnot_circuit

    ai_permutations_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2),
            AIPermutationSynthesis(
                backend_name="ibm_brisbane", replace_only_if_better=False
            ),
        ]
    )

    synthesized_circuit = ai_permutations_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
    assert synthesized_circuit == original_circuit
    assert "Using the synthesized circuit" in caplog.text


def test_ai_local_permutation_synthesis_with_backend_name(permutation_circuit_brisbane):
    original_circuit = permutation_circuit_brisbane

    ai_permutations_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=6),
            AIPermutationSynthesis(backend_name="ibm_brisbane"),
        ]
    )

    synthesized_circuit = ai_permutations_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)


def test_ai_local_permutation_synthesis_with_backend(
    permutation_circuit_brisbane, brisbane_backend
):
    original_circuit = permutation_circuit_brisbane

    ai_permutations_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=6),
            AIPermutationSynthesis(backend=brisbane_backend),
        ]
    )

    synthesized_circuit = ai_permutations_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)


def test_ai_local_permutation_synthesis_with_coupling_map(
    permutation_circuit_brisbane, brisbane_backend
):
    original_circuit = permutation_circuit_brisbane

    backend_coupling_map = brisbane_backend.coupling_map

    ai_permutations_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=6),
            AIPermutationSynthesis(coupling_map=backend_coupling_map),
        ]
    )

    synthesized_circuit = ai_permutations_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
