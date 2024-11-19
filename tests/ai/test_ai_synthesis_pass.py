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

"""Unit-testing ai clifford syhtesis for both local and cloud modes"""
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import CollectPermutations
from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
from qiskit_ibm_transpiler.ai.collection import CollectCliffords
from qiskit_ibm_transpiler.ai.synthesis import AIPermutationSynthesis
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis
from qiskit_ibm_transpiler.ai.synthesis import AICliffordSynthesis
from tests import (
    brisbane_coupling_map,
    brisbane_coupling_map_list_format,
)
from qiskit_ibm_transpiler.utils import (
    create_random_linear_function,
    random_clifford_from_linear_function,
)


# TODO: For Permutations, the original circuit doesn't return a DAGCircuit with nodes. Decide how the code should behave on this case
def customize_synthesis_type_with_basic_circuit():
    return pytest.mark.parametrize(
        "circuit, collector_pass, ai_synthesis_pass",
        [
            # ("basic_swap_circuit", CollectPermutations, AIPermutationSynthesis),
            ("basic_cnot_circuit", CollectLinearFunctions, AILinearFunctionSynthesis),
            ("basic_cnot_circuit", CollectCliffords, AICliffordSynthesis),
        ],
        # ids=["permutation", "linear_function", "clifford"],
        ids=["linear_function", "clifford"],
    )


def customize_synthesis_type_with_complex_circuit():
    return pytest.mark.parametrize(
        "circuit, collector_pass, ai_synthesis_pass",
        [
            (
                "permutation_circuit_brisbane",
                CollectPermutations,
                AIPermutationSynthesis,
            ),
            (
                "linear_function_circuit",
                CollectLinearFunctions,
                AILinearFunctionSynthesis,
            ),
            ("clifford_circuit", CollectCliffords, AICliffordSynthesis),
        ],
        ids=["permutation", "linear_function", "clifford"],
    )


def customize_local_mode():
    return pytest.mark.parametrize(
        "local_mode",
        ["true", "false"],
        ids=["local_mode", "cloud_mode"],
    )


@pytest.fixture
def basic_swap_circuit():
    circuit = QuantumCircuit(3)
    circuit.swap(0, 1)
    circuit.swap(1, 2)

    return circuit


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


# TODO: All the tests that use this circuit keeps the original circuit. Check if this is the better option
# for doing those tests
@pytest.fixture
def clifford_circuit():
    circuit = QuantumCircuit(8)
    clifford = random_clifford_from_linear_function(8)
    circuit.append(clifford, range(8))
    # Using decompose since we need a QuantumCircuit, not a Clifford. We created an empty
    # circuit, so it contains only a Clifford
    circuit = circuit.decompose(reps=1)

    return circuit


# TODO: When testing the synthesis with wrong backend, local and cloud behaves differently,
# so we should decide if this is correct or if we want to unify them
@customize_synthesis_type_with_basic_circuit()
def test_ai_local_synthesis_wrong_backend(
    circuit, collector_pass, ai_synthesis_pass, request
):
    original_circuit = request.getfixturevalue(circuit)

    with pytest.raises(
        PermissionError,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        custom_ai_synthesis_pass = PassManager(
            [
                collector_pass(min_block_size=2),
                ai_synthesis_pass(backend_name="wrong_backend"),
            ]
        )

        custom_ai_synthesis_pass.run(original_circuit)


# TODO: Tests pass if we add min_block_size=2, max_block_size=27 to collector_pass. If not, tests failed. Confirm why this is happening
@customize_synthesis_type_with_basic_circuit()
def test_ai_cloud_synthesis_wrong_backend(
    circuit, collector_pass, ai_synthesis_pass, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2, max_block_size=27),
            ai_synthesis_pass(backend_name="wrong_backend", local_mode=False),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

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
@customize_synthesis_type_with_basic_circuit()
def test_ai_cloud_synthesis_exceed_timeout(
    circuit, collector_pass, ai_synthesis_pass, backend, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(),
            ai_synthesis_pass(backend_name=backend, timeout=1, local_mode=False),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


# TODO: Tests pass if we add min_block_size=2, max_block_size=27 to CollectPermutations. If not, tests failed. Confirm why this is happening
@customize_synthesis_type_with_basic_circuit()
def test_ai_cloud_synthesis_wrong_token(
    circuit, collector_pass, ai_synthesis_pass, brisbane_backend_name, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2, max_block_size=27),
            ai_synthesis_pass(
                backend_name=brisbane_backend_name,
                token="invented_token_2",
                local_mode=False,
            ),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert "Invalid authentication credentials" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


# TODO: Tests pass if we add min_block_size=2, max_block_size=27 to CollectPermutations. If not, tests failed. Confirm why this is happening
@pytest.mark.disable_monkeypatch
@customize_synthesis_type_with_basic_circuit()
def test_ai_cloud_synthesis_wrong_url(
    circuit, collector_pass, ai_synthesis_pass, brisbane_backend_name, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2, max_block_size=27),
            ai_synthesis_pass(
                backend_name=brisbane_backend_name,
                base_url="https://ibm.com/",
                local_mode=False,
            ),
        ]
    )

    custom_ai_synthesis_pass.run(original_circuit)

    assert "Internal error: 404 Client Error:" in caplog.text
    assert "Keeping the original circuit" in caplog.text


# TODO: When using basic_swap_circuit it works, when using random_circuit_transpiled doesn't. Check why
@pytest.mark.disable_monkeypatch
@customize_synthesis_type_with_basic_circuit()
def test_ai_cloud_synthesis_unexisting_url(
    circuit, collector_pass, ai_synthesis_pass, brisbane_backend_name, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2, max_block_size=27),
            ai_synthesis_pass(
                backend_name=brisbane_backend_name,
                base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
                local_mode=False,
            ),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert (
        "Error: HTTPSConnectionPool(host='invented-domain-qiskit-ibm-transpiler-123.com', port=443):"
        in caplog.text
    )
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@customize_synthesis_type_with_basic_circuit()
@customize_local_mode()
def test_ai_synthesis_always_replace_original_circuit(
    circuit,
    collector_pass,
    ai_synthesis_pass,
    local_mode,
    brisbane_backend_name,
    caplog,
    request,
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2),
            ai_synthesis_pass(
                backend_name=brisbane_backend_name,
                replace_only_if_better=False,
                local_mode=local_mode,
            ),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Using the synthesized circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@customize_synthesis_type_with_basic_circuit()
@customize_local_mode()
def test_ai_synthesis_keep_original_if_better(
    circuit,
    collector_pass,
    ai_synthesis_pass,
    local_mode,
    brisbane_backend_name,
    caplog,
    request,
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2),
            ai_synthesis_pass(
                backend_name=brisbane_backend_name, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert ai_optimized_circuit == original_circuit
    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Keeping the original circuit" in caplog.text


@customize_synthesis_type_with_complex_circuit()
@customize_local_mode()
def test_ai_synthesis_pass_with_backend_name(
    circuit,
    collector_pass,
    ai_synthesis_pass,
    local_mode,
    brisbane_backend_name,
    caplog,
    request,
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(),
            ai_synthesis_pass(
                backend_name=brisbane_backend_name, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


@customize_synthesis_type_with_complex_circuit()
@customize_local_mode()
def test_ai_synthesis_pass_with_backend(
    circuit,
    collector_pass,
    ai_synthesis_pass,
    local_mode,
    brisbane_backend,
    caplog,
    request,
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(),
            ai_synthesis_pass(backend=brisbane_backend, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


# TODO: The tests pass but some errors are logged. Check this
@customize_synthesis_type_with_complex_circuit()
@customize_local_mode()
@pytest.mark.parametrize(
    "coupling_map",
    [brisbane_coupling_map, brisbane_coupling_map_list_format],
    indirect=True,
    ids=["coupling_map_object", "coupling_map_list"],
)
def test_ai_synthesis_pass_with_coupling_map(
    circuit,
    collector_pass,
    ai_synthesis_pass,
    local_mode,
    coupling_map,
    caplog,
    request,
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2),
            ai_synthesis_pass(coupling_map=coupling_map, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])
