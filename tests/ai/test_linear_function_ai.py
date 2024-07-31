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


def test_linear_function_wrong_backend(random_circuit_transpiled, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend_name="wrong_backend"),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert (
        "User doesn't have access to the specified backend: wrong_backend"
        in caplog.text
    )
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.skip(
    reason="Unreliable. It passes most of the times with the timeout of 1 second for the current circuits used"
)
def test_linear_function_exceed_timeout(random_circuit_transpiled, backend, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend_name=backend, timeout=1),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_linear_function_wrong_token(random_circuit_transpiled, backend, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend_name=backend, token="invented_token_2"),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert "Invalid authentication credentials" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.disable_monkeypatch
def test_linear_function_wrong_url(random_circuit_transpiled, backend):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(
                backend_name=backend, base_url="https://ibm.com/"
            ),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Expecting value: line 1 column 1 (char 0)" in str(e)
        assert type(e).__name__ == "JSONDecodeError"


@pytest.mark.disable_monkeypatch
def test_linear_function_unexisting_url(random_circuit_transpiled, backend, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(
                backend_name=backend,
                base_url="https://invented-domain-qiskit-transpiler-service-123.com/",
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert (
        "Error: HTTPSConnectionPool(host='invented-domain-qiskit-transpiler-service-123.com', port=443):"
        in caplog.text
    )
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


def test_linear_function_pass(random_circuit_transpiled, backend, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend_name=backend),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
