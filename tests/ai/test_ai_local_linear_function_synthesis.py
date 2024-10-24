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


def test_ai_local_linear_function_synthesis_wrong_backend():
    original_circuit = QuantumCircuit(3)
    original_circuit.cx(0, 1)
    original_circuit.cx(1, 2)

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend_name="wrong_backend"),
        ]
    )

    with pytest.raises(
        PermissionError,
        match=r"ERROR. Backend not supported \(\w+\)",
    ):
        ai_linear_functions_synthesis_pass.run(original_circuit)


def test_ai_local_linear_function_synthesis_returns_original_circuit(caplog):
    # When the original circuit is better than the synthetized one, we keep the original

    original_circuit = QuantumCircuit(3)
    original_circuit.cx(0, 1)
    original_circuit.cx(1, 2)

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend_name="ibm_brisbane"),
        ]
    )

    synthetized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthetized_circuit, QuantumCircuit)
    assert synthetized_circuit == original_circuit
    assert "Keeping the original circuit" in caplog.text


def test_ai_local_linear_function_synthesis_dont_returns_original_circuit(caplog):
    # When the original circuit is better than the synthetized one,
    # but replace_only_if_better is False, we return the synthetized circuit

    original_circuit = QuantumCircuit(3)
    original_circuit.cx(0, 1)
    original_circuit.cx(1, 2)

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(
                backend_name="ibm_brisbane", replace_only_if_better=False
            ),
        ]
    )

    synthetized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthetized_circuit, QuantumCircuit)
    assert synthetized_circuit == original_circuit
    assert "Using the synthesized circuit" in caplog.text


def test_ai_local_linear_function_synthesis(caplog):
    original_circuit = QuantumCircuit(8)
    linear_function = create_random_linear_function(8)
    original_circuit.append(linear_function, range(8))
    # Using decompose since we need a QuantumCircuit, not a LinearFunction. We created original_circuit
    # empty, so it contains only a LinearFunction
    original_circuit = original_circuit.decompose(reps=1)

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend_name="ibm_brisbane"),
        ]
    )

    synthetized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthetized_circuit, QuantumCircuit)
