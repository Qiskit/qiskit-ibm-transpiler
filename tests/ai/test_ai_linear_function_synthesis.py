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

"""Unit-testing ai Linear Function syhtesis for both local and cloud modes"""
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis
from qiskit_ibm_transpiler.utils import create_random_linear_function


# TODO: All the tests that use this circuit keeps the original circuit. Check if this is the better option
# for doing those tests
@pytest.fixture
def linear_function_circuit():
    circuit = QuantumCircuit(8)
    linear_function = create_random_linear_function(8)
    circuit.append(linear_function, range(8))
    # Using decompose since we need a QuantumCircuit, not a LinearFunction. We created an empty
    # circuit, so it contains only a LinearFunction
    circuit = circuit.decompose(reps=1)

    return circuit


# TODO: When testing the linear function synthesis with wrong backend, local and cloud behaves differently,
# so we should decide if this is correct or if we want to unify them
def test_ai_local_linear_function_synthesis_wrong_backend(basic_cnot_circuit):
    original_circuit = basic_cnot_circuit

    with pytest.raises(
        PermissionError,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        ai_linear_functions_synthesis_pass = PassManager(
            [
                CollectLinearFunctions(min_block_size=2),
                AILinearFunctionSynthesis(backend_name="wrong_backend"),
            ]
        )

        ai_linear_functions_synthesis_pass.run(original_circuit)


def test_ai_cloud_linear_function_synthesis_wrong_backend(
    random_circuit_transpiled, caplog
):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend_name="wrong_backend", local_mode=False),
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
def test_ai_cloud_linear_function_synthesis_exceed_timeout(
    random_circuit_transpiled, backend, caplog
):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(
                backend_name=backend, timeout=1, local_mode=False
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.skip(
    reason="Unreliable many times. We'll research why it fails sporadically"
)
def test_ai_cloud_linear_function_synthesis_wrong_token(
    random_circuit_transpiled, backend, caplog
):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(
                backend_name=backend, token="invented_token_2", local_mode=False
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert "Invalid authentication credentials" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.skip(
    reason="Unreliable many times. We'll research why it fails sporadically"
)
@pytest.mark.disable_monkeypatch
def test_ai_cloud_linear_function_synthesis_wrong_url(
    random_circuit_transpiled, backend
):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(
                backend_name=backend, base_url="https://ibm.com/", local_mode=False
            ),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Expecting value: line 1 column 1 (char 0)" in str(e)
        assert type(e).__name__ == "JSONDecodeError"


@pytest.mark.skip(
    reason="Unreliable many times. We'll research why it fails sporadically"
)
@pytest.mark.disable_monkeypatch
def test_ai_cloud_linear_function_synthesis_unexisting_url(
    random_circuit_transpiled, backend, caplog
):
    ai_optimize_lf = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(
                backend_name=backend,
                base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
                local_mode=False,
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert (
        "Error: HTTPSConnectionPool(host='invented-domain-qiskit-ibm-transpiler-123.com', port=443):"
        in caplog.text
    )
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


# TODO: Tests pass if we add min_block_size=2 to CollectLinearFunctions. If not, tests failed. Confirm why this is happening
@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_linear_function_synthesis_always_replace_original_circuit(
    basic_cnot_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = basic_cnot_circuit

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(
                backend_name=brisbane_backend_name,
                replace_only_if_better=False,
                local_mode=local_mode,
            ),
        ]
    )

    ai_optimized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Using the synthesized circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_linear_function_synthesis_keep_original_if_better(
    basic_cnot_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = basic_cnot_circuit

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(
                backend_name=brisbane_backend_name, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert ai_optimized_circuit == original_circuit
    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Keeping the original circuit" in caplog.text


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_linear_function_synthesis_pass_with_backend_name(
    linear_function_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = linear_function_circuit

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(
                backend_name=brisbane_backend_name, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_linear_function_synthesis_pass_with_backend(
    linear_function_circuit, brisbane_backend, caplog, local_mode
):
    original_circuit = linear_function_circuit

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(),
            AILinearFunctionSynthesis(backend=brisbane_backend, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_linear_function_synthesis_pass_with_coupling_map(
    linear_function_circuit, brisbane_backend, caplog, local_mode
):
    original_circuit = linear_function_circuit

    backend_coupling_map = brisbane_backend.coupling_map

    ai_linear_functions_synthesis_pass = PassManager(
        [
            CollectLinearFunctions(min_block_size=2),
            AILinearFunctionSynthesis(
                coupling_map=backend_coupling_map, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = ai_linear_functions_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])
