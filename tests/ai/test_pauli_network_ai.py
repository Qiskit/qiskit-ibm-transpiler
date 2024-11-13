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

"""Unit-testing pauli_network_ai"""
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import CollectPauliNetworks
from qiskit_ibm_transpiler.ai.synthesis import AIPauliNetworkSynthesis


def test_pauli_network_wrong_backend(random_circuit_transpiled, caplog):
    ai_optimize_pauli = PassManager(
        [
            CollectPauliNetworks(),
            AIPauliNetworkSynthesis(backend_name="wrong_backend", local_mode=False),
        ]
    )
    ai_optimized_circuit = ai_optimize_pauli.run(random_circuit_transpiled)
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
def test_pauli_network_exceed_timeout(random_circuit_transpiled, backend, caplog):
    ai_optimize_cliff = PassManager(
        [
            CollectPauliNetworks(),
            AIPauliNetworkSynthesis(backend_name=backend, timeout=1, local_mode=False),
        ]
    )
    ai_optimized_circuit = ai_optimize_cliff.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_pauli_network_wrong_token(random_circuit_transpiled, backend, caplog):
    ai_optimize_cliff = PassManager(
        [
            CollectPauliNetworks(),
            AIPauliNetworkSynthesis(
                backend_name=backend, token="invented_token_2", local_mode=False
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_cliff.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert "Invalid authentication credentials" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


@pytest.mark.disable_monkeypatch
def test_pauli_network_wrong_url(random_circuit_transpiled, backend, caplog):
    ai_optimize_cliff = PassManager(
        [
            CollectPauliNetworks(),
            AIPauliNetworkSynthesis(
                backend_name=backend, base_url="https://ibm.com/", local_mode=False
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_cliff.run(random_circuit_transpiled)
    assert "Internal error: 404 Client Error:" in caplog.text
    assert "Keeping the original circuit" in caplog.text


@pytest.mark.disable_monkeypatch
def test_pauli_network_unexisting_url(random_circuit_transpiled, backend, caplog):
    ai_optimize_cliff = PassManager(
        [
            CollectPauliNetworks(),
            AIPauliNetworkSynthesis(
                backend_name=backend,
                base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
                local_mode=False,
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_cliff.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert (
        "Error: HTTPSConnectionPool(host='invented-domain-qiskit-ibm-transpiler-123.com', port=443):"
        in caplog.text
    )
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_pauli_network_function(random_pauli_circuit_transpiled, backend_27q, caplog):
    ai_optimize_cliff = PassManager(
        [
            CollectPauliNetworks(),
            AIPauliNetworkSynthesis(backend_name=backend_27q, local_mode=False),
        ]
    )
    from qiskit import qasm2

    with open("pauli_circuit_2.qasm", "w") as f:
        qasm2.dump(random_pauli_circuit_transpiled, f)
    ai_optimized_circuit = ai_optimize_cliff.run(random_pauli_circuit_transpiled)
    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert "Using the synthesized circuit" in caplog.text


# TODO: Look for a better way to parametrize coupling maps
@pytest.mark.parametrize(
    "use_coupling_map_as_list", [True, False], ids=["as_list", "as_object"]
)
def test_pauli_network_function_with_coupling_map(
    random_pauli_circuit_transpiled,
    backend_27q,
    cmap_backend,
    use_coupling_map_as_list,
    caplog,
):
    coupling_map_to_send = (
        list(cmap_backend[backend_27q].get_edges())
        if use_coupling_map_as_list
        else cmap_backend[backend_27q]
    )
    ai_optimize_cliff = PassManager(
        [
            CollectPauliNetworks(),
            AIPauliNetworkSynthesis(
                coupling_map=coupling_map_to_send, local_mode=False
            ),
        ]
    )
    ai_optimized_circuit = ai_optimize_cliff.run(random_pauli_circuit_transpiled)
    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert "Using the synthesized circuit" in caplog.text
