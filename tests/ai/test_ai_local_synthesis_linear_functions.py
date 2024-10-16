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
from qiskit.quantum_info import Operator
import numpy as np

from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis


# def test_linear_function_wrong_backend(random_circuit_transpiled, caplog):
#     ai_optimize_lf = PassManager(
#         [
#             CollectLinearFunctions(),
#             AILinearFunctionSynthesis(backend_name="wrong_backend"),
#         ]
#     )
#     ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
#     assert "couldn't synthesize the circuit" in caplog.text
#     assert "Keeping the original circuit" in caplog.text
#     assert (
#         "User doesn't have access to the specified backend: wrong_backend"
#         in caplog.text
#     )
#     assert isinstance(ai_optimized_circuit, QuantumCircuit)


# @pytest.mark.skip(
#     reason="Unreliable. It passes most of the times with the timeout of 1 second for the current circuits used"
# )
# def test_linear_function_exceed_timeout(random_circuit_transpiled, backend, caplog):
#     ai_optimize_lf = PassManager(
#         [
#             CollectLinearFunctions(),
#             AILinearFunctionSynthesis(backend_name=backend, timeout=1),
#         ]
#     )
#     ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
#     assert "couldn't synthesize the circuit" in caplog.text
#     assert "Keeping the original circuit" in caplog.text
#     assert isinstance(ai_optimized_circuit, QuantumCircuit)


# @pytest.mark.skip(
#     reason="Unreliable many times. We'll research why it fails sporadically"
# )
# def test_linear_function_wrong_token(random_circuit_transpiled, backend, caplog):
#     ai_optimize_lf = PassManager(
#         [
#             CollectLinearFunctions(),
#             AILinearFunctionSynthesis(backend_name=backend, token="invented_token_2"),
#         ]
#     )
#     ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
#     assert "couldn't synthesize the circuit" in caplog.text
#     assert "Keeping the original circuit" in caplog.text
#     assert "Invalid authentication credentials" in caplog.text
#     assert isinstance(ai_optimized_circuit, QuantumCircuit)


# @pytest.mark.skip(
#     reason="Unreliable many times. We'll research why it fails sporadically"
# )
# @pytest.mark.disable_monkeypatch
# def test_linear_function_wrong_url(random_circuit_transpiled, backend):
#     ai_optimize_lf = PassManager(
#         [
#             CollectLinearFunctions(),
#             AILinearFunctionSynthesis(
#                 backend_name=backend, base_url="https://ibm.com/"
#             ),
#         ]
#     )
#     try:
#         ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
#         pytest.fail("Error expected")
#     except Exception as e:
#         assert "Expecting value: line 1 column 1 (char 0)" in str(e)
#         assert type(e).__name__ == "JSONDecodeError"


# @pytest.mark.skip(
#     reason="Unreliable many times. We'll research why it fails sporadically"
# )
# @pytest.mark.disable_monkeypatch
# def test_linear_function_unexisting_url(random_circuit_transpiled, backend, caplog):
#     ai_optimize_lf = PassManager(
#         [
#             CollectLinearFunctions(),
#             AILinearFunctionSynthesis(
#                 backend_name=backend,
#                 base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
#             ),
#         ]
#     )
#     ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
#     assert "couldn't synthesize the circuit" in caplog.text
#     assert "Keeping the original circuit" in caplog.text
#     assert (
#         "Error: HTTPSConnectionPool(host='invented-domain-qiskit-ibm-transpiler-123.com', port=443):"
#         in caplog.text
#     )
#     assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_linear_function_synthesis_pass(random_circuit_transpiled):
    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend_name="ibm_montecarlo"),
        ]
    )

    synthetized_circuit = ai_linear_functions_synthesis_pass.run(
        random_circuit_transpiled
    )

    assert isinstance(synthetized_circuit, QuantumCircuit)


def test_linear_function_synthesis_returns_original_circuit(caplog):
    # When the original circuit is better than the synthetized one, we keep the original

    original_circuit = QuantumCircuit(3)
    original_circuit.cx(0, 1)
    original_circuit.cx(1, 2)

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(backend_name="ibm_montecarlo"),
        ]
    )

    synthetized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthetized_circuit, QuantumCircuit)
    assert synthetized_circuit == original_circuit
    assert "Keeping the original circuit" in caplog.text


def test_linear_function_synthesis_dont_returns_original_circuit(caplog):
    # When the original circuit is better than the synthetized one,
    # but replace_only_if_better is False, we return the synthetized circuit

    original_circuit = QuantumCircuit(3)
    original_circuit.cx(0, 1)
    original_circuit.cx(1, 2)

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(
                backend_name="ibm_montecarlo", replace_only_if_better=False
            ),
        ]
    )

    synthetized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(synthetized_circuit, QuantumCircuit)
    assert synthetized_circuit == original_circuit
    assert "Using the synthesized circuit" in caplog.text
