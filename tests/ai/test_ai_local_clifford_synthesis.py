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

"""Testing AI local synthesis for cliffords"""
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import CollectCliffords
from qiskit_ibm_transpiler.utils import random_clifford_from_linear_function
from qiskit_ibm_transpiler.ai.synthesis import AICliffordSynthesis
from qiskit_ibm_runtime import QiskitRuntimeService


@pytest.fixture
def basic_cnot_circuit():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    return circuit


@pytest.fixture
def clifford_circuit():
    circuit = QuantumCircuit(8)
    clifford = random_clifford_from_linear_function(8)
    circuit.append(clifford, range(8))
    # Using decompose since we need a QuantumCircuit, not a Clifford. We created an empty
    # circuit, so it contains only a Clifford
    circuit = circuit.decompose(reps=1)

    return circuit


@pytest.fixture
def brisbane_backend():
    backend = QiskitRuntimeService().backend("ibm_brisbane")

    return backend


def test_ai_local_clifford_synthesis_wrong_backend(basic_cnot_circuit):
    original_circuit = basic_cnot_circuit

    with pytest.raises(
        PermissionError,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        ai_cliffords_synthesis_pass = PassManager(
            [
                CollectCliffords(min_block_size=2),
                AICliffordSynthesis(backend_name="wrong_backend"),
            ]
        )

        ai_cliffords_synthesis_pass.run(original_circuit)


def test_ai_local_clifford_synthesis_returns_original_circuit(
    basic_cnot_circuit, caplog
):
    # When the original circuit is better than the synthesized one, we keep the original

    original_circuit = basic_cnot_circuit

    ai_cliffords_synthesis_pass = PassManager(
        [
            CollectCliffords(min_block_size=2),
            AICliffordSynthesis(backend_name="ibm_brisbane"),
        ]
    )

    synthesized_circuit = ai_cliffords_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
    assert synthesized_circuit == original_circuit
    assert "Keeping the original circuit" in caplog.text


def test_ai_local_clifford_synthesis_dont_returns_original_circuit(
    basic_cnot_circuit, caplog
):
    # When the original circuit is better than the synthesized one,
    # but replace_only_if_better is False, we return the synthesized circuit

    original_circuit = basic_cnot_circuit

    ai_cliffords_synthesis_pass = PassManager(
        [
            CollectCliffords(min_block_size=2),
            AICliffordSynthesis(
                backend_name="ibm_brisbane", replace_only_if_better=False
            ),
        ]
    )

    synthesized_circuit = ai_cliffords_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
    assert synthesized_circuit == original_circuit
    assert "Using the synthesized circuit" in caplog.text


def test_ai_local_clifford_synthesis_with_backend_name(clifford_circuit):
    original_circuit = clifford_circuit

    ai_cliffords_synthesis_pass = PassManager(
        [
            CollectCliffords(min_block_size=2),
            AICliffordSynthesis(backend_name="ibm_brisbane"),
        ]
    )

    synthesized_circuit = ai_cliffords_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)


def test_ai_local_clifford_synthesis_with_backend(clifford_circuit, brisbane_backend):
    original_circuit = clifford_circuit

    ai_cliffords_synthesis_pass = PassManager(
        [
            CollectCliffords(min_block_size=2),
            AICliffordSynthesis(backend=brisbane_backend),
        ]
    )

    synthesized_circuit = ai_cliffords_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)


def test_ai_local_clifford_synthesis_with_coupling_map(
    clifford_circuit, brisbane_backend
):
    original_circuit = clifford_circuit

    backend_coupling_map = brisbane_backend.coupling_map

    ai_cliffords_synthesis_pass = PassManager(
        [
            CollectCliffords(min_block_size=2),
            AICliffordSynthesis(coupling_map=backend_coupling_map),
        ]
    )

    synthesized_circuit = ai_cliffords_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
