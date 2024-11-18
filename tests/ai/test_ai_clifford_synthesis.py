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

from qiskit_ibm_transpiler.ai.collection import CollectCliffords
from qiskit_ibm_transpiler.ai.synthesis import AICliffordSynthesis
from qiskit_ibm_transpiler.utils import random_clifford_from_linear_function
from tests import brisbane_coupling_map, brisbane_coupling_map_list_format


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


# TODO: When testing the clifford synthesis with wrong backend, local and cloud behaves differently,
# so we should decide if this is correct or if we want to unify them
def test_ai_local_clifford_synthesis_wrong_backend(basic_cnot_circuit):
    original_circuit = basic_cnot_circuit

    with pytest.raises(
        PermissionError,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        ai_clifford_synthesis_pass = PassManager(
            [
                CollectCliffords(min_block_size=2),
                AICliffordSynthesis(backend_name="wrong_backend"),
            ]
        )

        ai_clifford_synthesis_pass.run(original_circuit)


def test_ai_cloud_clifford_synthesis_wrong_backend(random_circuit_transpiled, caplog):
    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(backend_name="wrong_backend", local_mode=False),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(random_circuit_transpiled)

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
def test_ai_cloud_clifford_synthesis_exceed_timeout(
    random_circuit_transpiled, backend, caplog
):
    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(backend_name=backend, timeout=1, local_mode=False),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(random_circuit_transpiled)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_ai_cloud_clifford_synthesis_wrong_token(
    random_circuit_transpiled, brisbane_backend_name, caplog
):
    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(
                backend_name=brisbane_backend_name,
                token="invented_token_2",
                local_mode=False,
            ),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(random_circuit_transpiled)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert "Invalid authentication credentials" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.disable_monkeypatch
def test_ai_cloud_clifford_synthesis_wrong_url(
    random_circuit_transpiled, backend, caplog
):
    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(
                backend_name=backend, base_url="https://ibm.com/", local_mode=False
            ),
        ]
    )

    ai_clifford_synthesis_pass.run(random_circuit_transpiled)

    assert "Internal error: 404 Client Error:" in caplog.text
    assert "Keeping the original circuit" in caplog.text


@pytest.mark.disable_monkeypatch
def test_ai_cloud_clifford_synthesis_unexisting_url(
    random_circuit_transpiled, backend, caplog
):
    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(
                backend_name=backend,
                base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
                local_mode=False,
            ),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(random_circuit_transpiled)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert (
        "Error: HTTPSConnectionPool(host='invented-domain-qiskit-ibm-transpiler-123.com', port=443):"
        in caplog.text
    )
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


# TODO: Tests pass if we add min_block_size=2 to CollectCliffords. If not, tests failed. Confirm why this is happening
@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_clifford_synthesis_always_replace_original_circuit(
    basic_cnot_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = basic_cnot_circuit

    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(min_block_size=2),
            AICliffordSynthesis(
                backend_name=brisbane_backend_name,
                replace_only_if_better=False,
                local_mode=local_mode,
            ),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(original_circuit)

    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Using the synthesized circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_clifford_synthesis_keep_original_if_better(
    basic_cnot_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = basic_cnot_circuit

    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(min_block_size=2),
            AICliffordSynthesis(
                backend_name=brisbane_backend_name, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert ai_optimized_circuit == original_circuit
    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Keeping the original circuit" in caplog.text


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_clifford_synthesis_pass_with_backend_name(
    clifford_circuit, brisbane_backend_name, caplog, local_mode
):
    original_circuit = clifford_circuit

    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(
                backend_name=brisbane_backend_name, local_mode=local_mode
            ),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


@pytest.mark.parametrize(
    "local_mode",
    [None, "true", "false"],
    ids=["default_local_mode", "specify_local_mode", "specify_cloud_mode"],
)
def test_ai_clifford_synthesis_pass_with_backend(
    clifford_circuit, brisbane_backend, caplog, local_mode
):
    original_circuit = clifford_circuit

    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(backend=brisbane_backend, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


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
def test_ai_clifford_synthesis_pass_with_coupling_map(
    clifford_circuit, caplog, local_mode, coupling_map
):
    original_circuit = clifford_circuit

    ai_clifford_synthesis_pass = PassManager(
        [
            CollectCliffords(min_block_size=2),
            AICliffordSynthesis(coupling_map=coupling_map, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = ai_clifford_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])
