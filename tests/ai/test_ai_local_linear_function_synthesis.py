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

from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
from qiskit_ibm_transpiler.utils import create_random_linear_function, get_metrics
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis
from qiskit_ibm_runtime import QiskitRuntimeService


@pytest.fixture
def basic_cnot_circuit():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    return circuit


@pytest.fixture
def linear_function_circuit():
    circuit = QuantumCircuit(8)
    linear_function = create_random_linear_function(8)
    circuit.append(linear_function, range(8))
    # Using decompose since we need a QuantumCircuit, not a LinearFunction. We created an empty
    # circuit, so it contains only a LinearFunction
    circuit = circuit.decompose(reps=1)

    return circuit


@pytest.fixture
def brisbane_backend():
    backend = QiskitRuntimeService().backend("ibm_brisbane")

    return backend


def test_ai_local_linear_function_synthesis_wrong_backend(basic_cnot_circuit):
    original_circuit = basic_cnot_circuit

    with pytest.raises(
        PermissionError,
        match=r"ERROR. Backend not supported \(\w+\)",
    ):
        ai_linear_functions_synthesis_pass = PassManager(
            [
                CollectLinearFunctions(min_block_size=2),
                AILinearFunctionSynthesis(backend_name="wrong_backend"),
            ]
        )

        ai_linear_functions_synthesis_pass.run(original_circuit)


def test_ai_local_linear_function_synthesis_returns_original_circuit(
    basic_cnot_circuit, caplog
):
    # When the original circuit is better than the synthesized one, we keep the original

    original_circuit = basic_cnot_circuit

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend_name="ibm_brisbane"),
        ]
    )

    synthesized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
    assert synthesized_circuit == original_circuit
    assert "Keeping the original circuit" in caplog.text


def test_ai_local_linear_function_synthesis_dont_returns_original_circuit(
    basic_cnot_circuit, caplog
):
    # When the original circuit is better than the synthesized one,
    # but replace_only_if_better is False, we return the synthesized circuit

    original_circuit = basic_cnot_circuit

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(
                backend_name="ibm_brisbane", replace_only_if_better=False
            ),
        ]
    )

    synthesized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
    assert synthesized_circuit == original_circuit
    assert "Using the synthesized circuit" in caplog.text


def test_ai_local_linear_function_synthesis_with_backend_name(linear_function_circuit):
    original_circuit = linear_function_circuit

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend_name="ibm_brisbane"),
        ]
    )

    synthesized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)


def test_ai_local_linear_function_synthesis_with_backend(
    linear_function_circuit, brisbane_backend
):
    original_circuit = linear_function_circuit

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend=brisbane_backend),
        ]
    )

    synthesized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)


def test_ai_local_linear_function_synthesis_with_coupling_map(
    linear_function_circuit, brisbane_backend
):
    original_circuit = linear_function_circuit

    backend_coupling_map = brisbane_backend.coupling_map

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(coupling_map=backend_coupling_map),
        ]
    )

    synthesized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthesized_circuit, QuantumCircuit)
