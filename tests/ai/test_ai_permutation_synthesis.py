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
from qiskit_ibm_transpiler.ai.synthesis import AIPermutationSynthesis
from tests import brisbane_coupling_map, brisbane_coupling_map_list_format


@pytest.fixture
def basic_cnot_circuit():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    return circuit


@pytest.fixture
def basic_swap_circuit():
    circuit = QuantumCircuit(3)
    circuit.swap(0, 1)
    circuit.swap(1, 2)

    return circuit


# TODO: When testing the permutation synthesis with wrong backend, local and cloud behaves differently,
# so we should decide if this is correct or if we want to unify them
def test_ai_local_permutation_synthesis_wrong_backend(basic_swap_circuit):
    original_circuit = basic_swap_circuit

    with pytest.raises(
        PermissionError,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        ai_permutation_synthesis_pass = PassManager(
            [
                CollectPermutations(min_block_size=2),
                AIPermutationSynthesis(backend_name="wrong_backend"),
            ]
        )

        ai_permutation_synthesis_pass.run(original_circuit)


# TODO: Tests pass if we add min_block_size=2, max_block_size=27 to CollectPermutations. If not, tests failed. Confirm why this is happening
def test_ai_cloud_permutation_synthesis_wrong_backend(basic_swap_circuit, caplog):
    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(backend_name="wrong_backend", local_mode=False),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(basic_swap_circuit)

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
def test_ai_cloud_permutation_synthesis_exceed_timeout(
    basic_swap_circuit, backend, caplog
):
    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(),
            AIPermutationSynthesis(backend_name=backend, timeout=1, local_mode=False),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(basic_swap_circuit)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


# TODO: Tests pass if we add min_block_size=2, max_block_size=27 to CollectPermutations. If not, tests failed. Confirm why this is happening
def test_ai_cloud_permutation_synthesis_wrong_token(
    basic_swap_circuit, brisbane_backend_name, caplog
):
    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(
                backend_name=brisbane_backend_name,
                token="invented_token_2",
                local_mode=False,
            ),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(basic_swap_circuit)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert "Invalid authentication credentials" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


# TODO: Tests pass if we add min_block_size=2, max_block_size=27 to CollectPermutations. If not, tests failed. Confirm why this is happening
@pytest.mark.disable_monkeypatch
def test_ai_cloud_permutation_synthesis_wrong_url(basic_swap_circuit, backend, caplog):
    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(
                backend_name=backend, base_url="https://ibm.com/", local_mode=False
            ),
        ]
    )

    ai_permutation_synthesis_pass.run(basic_swap_circuit)

    assert "Internal error: 404 Client Error:" in caplog.text
    assert "Keeping the original circuit" in caplog.text


# TODO: When using basic_swap_circuit it works, when using random_circuit_transpiled doesn't. Check why
@pytest.mark.disable_monkeypatch
def test_ai_cloud_permutation_synthesis_unexisting_url(
    basic_swap_circuit, backend, caplog
):
    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(
                backend_name=backend,
                base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
                local_mode=False,
            ),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(basic_swap_circuit)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert (
        "Error: HTTPSConnectionPool(host='invented-domain-qiskit-ibm-transpiler-123.com', port=443):"
        in caplog.text
    )
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.skip(
    reason="The original circuit doesn't return a DAGCircuit with nodes. We are deciding how the code should behave on this case"
)
@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_permutation_synthesis_always_replace_original_circuit(
    basic_cnot_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = basic_cnot_circuit

    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2),
            AIPermutationSynthesis(
                backend_name=brisbane_backend_name,
                replace_only_if_better=False,
                local_mode=local_mode,
            ),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(original_circuit)

    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Using the synthesized circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.skip(
    reason="The original circuit doesn't return a DAGCircuit with nodes. We are deciding how the code should behave on this case"
)
@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_permutation_synthesis_keep_original_if_better(
    basic_cnot_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = basic_cnot_circuit

    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2),
            AIPermutationSynthesis(
                backend_name=brisbane_backend_name, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert ai_optimized_circuit == original_circuit
    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Keeping the original circuit" in caplog.text


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_permutation_synthesis_pass_with_backend_name(
    permutation_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = permutation_circuit

    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(),
            AIPermutationSynthesis(
                backend_name=brisbane_backend_name, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_permutation_synthesis_pass_with_backend(
    permutation_circuit, brisbane_backend, caplog, local_mode
):
    original_circuit = permutation_circuit

    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(),
            AIPermutationSynthesis(backend=brisbane_backend, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


# TODO: The tests pass but some errors are logged. Check this
@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
@pytest.mark.parametrize(
    "coupling_map",
    [brisbane_coupling_map, brisbane_coupling_map_list_format],
    indirect=True,
    ids=["coupling_map_object", "coupling_map_list"],
)
def test_ai_permutation_synthesis_pass_with_coupling_map(
    permutation_circuit, caplog, local_mode, coupling_map
):
    original_circuit = permutation_circuit

    ai_permutation_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2),
            AIPermutationSynthesis(coupling_map=coupling_map, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = ai_permutation_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])
