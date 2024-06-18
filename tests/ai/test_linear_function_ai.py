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

"""Unit-testing linear_function_ai"""
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_transpiler_service.ai.collection import CollectLinearFunctions
from qiskit_transpiler_service.ai.synthesis import AILinearFunctionSynthesis


def test_linear_function_pass(random_circuit_transpiled, backend, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend_name=backend),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_linear_function_wrong_backend(caplog):
    circuit = QuantumCircuit(10)
    for i in range(8):
        circuit.cx(i, i + 1)
    for i in range(8):
        circuit.h(i)
    for i in range(8):
        circuit.cx(i, i + 1)
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend_name="a_wrong_backend"),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(circuit)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_linear_always_replace(backend, caplog):
    orig_qc = QuantumCircuit(3)
    orig_qc.cx(0, 1)
    orig_qc.cx(1, 2)
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(
                backend_name=backend, replace_only_if_better=False
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(orig_qc)
    assert "Keeping the original circuit" not in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_linear_function_only_replace_if_better(backend, caplog):
    orig_qc = QuantumCircuit(3)
    orig_qc.cx(0, 1)
    orig_qc.cx(1, 2)
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend_name=backend),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(orig_qc)
    assert ai_optimized_circuit == orig_qc
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)
