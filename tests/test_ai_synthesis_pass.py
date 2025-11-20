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

"""Tests for the ai synthesis pass"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume, quantum_volume
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.collection import (
    CollectPauliNetworks,
    CollectPermutations,
)
from qiskit_ibm_transpiler.ai_passmanager import generate_ai_pass_manager
from tests.parametrize_functions import (
    parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass,
    parametrize_complex_circuit_collector_pass_and_ai_synthesis_pass,
    parametrize_coupling_map_format,
    parametrize_local_mode,
)


@pytest.mark.skip(
    reason="Unreliable. It passes most of the times with the timeout of 1 second for the current circuits used"
)
@parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass()
def test_ai_cloud_synthesis_exceed_timeout(
    circuit, collector_pass, ai_synthesis_pass, test_eagle_backend, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(),
            ai_synthesis_pass(backend=test_eagle_backend, timeout=1, local_mode=False),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


# On the collector_pass, the defautl min_block_size is 4, so if we use the circuits on
# parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass (less than 4 qubits in
# some cases), we should set min_block_size=2, max_block_size=27 as params in order to
# make the test pass
@pytest.mark.skip(reason="Disabling cloud tests for now")
@parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass()
def test_ai_cloud_synthesis_wrong_token(
    circuit, collector_pass, ai_synthesis_pass, test_eagle_backend, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2, max_block_size=27),
            ai_synthesis_pass(
                backend=test_eagle_backend,
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


# On the collector_pass, the defautl min_block_size is 4, so if we use the circuits on
# parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass (less than 4 qubits in
# some cases), we should set min_block_size=2, max_block_size=27 as params in order to
# make the test pass
@pytest.mark.skip(reason="Disabling cloud tests for now")
@pytest.mark.disable_monkeypatch
@parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass()
def test_ai_cloud_synthesis_wrong_url(
    circuit, collector_pass, ai_synthesis_pass, test_eagle_backend, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2, max_block_size=27),
            ai_synthesis_pass(
                backend=test_eagle_backend,
                base_url="https://ibm.com/",
                local_mode=False,
            ),
        ]
    )

    custom_ai_synthesis_pass.run(original_circuit)

    assert "Internal error: 404 Client Error:" in caplog.text
    assert "Keeping the original circuit" in caplog.text


# TODO: When using basic_swap_circuit it works, when using random_test_eagle_circuit_with_two_cliffords doesn't. Check why
@pytest.mark.skip(reason="Disabling cloud tests for now")
@pytest.mark.disable_monkeypatch
@parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass()
def test_ai_cloud_synthesis_unexisting_url(
    circuit, collector_pass, ai_synthesis_pass, test_eagle_backend, caplog, request
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2, max_block_size=27),
            ai_synthesis_pass(
                backend=test_eagle_backend,
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


# TODO: Review why if I remove max_block_size=27 from the collector_pass, Permutation
# test fails
@parametrize_complex_circuit_collector_pass_and_ai_synthesis_pass()
@parametrize_local_mode()
def test_ai_synthesis_always_replace_original_circuit(
    circuit,
    collector_pass,
    ai_synthesis_pass,
    local_mode,
    test_eagle_backend,
    caplog,
    request,
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(max_block_size=27),
            ai_synthesis_pass(
                backend=test_eagle_backend,
                replace_only_if_better=False,
                local_mode=local_mode,
            ),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Using the synthesized circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def flatten_opaque_with_circuit_params(qc: QuantumCircuit) -> QuantumCircuit:
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for instr, qargs, cargs in qc.data:
        if instr.definition is None and any(
            isinstance(p, QuantumCircuit) for p in instr.params
        ):
            # Find the circuit embedded in the parameters
            for param in instr.params:
                if isinstance(param, QuantumCircuit):
                    new_qc.compose(
                        param, qubits=[qc.qubits.index(q) for q in qargs], inplace=True
                    )
        else:
            new_qc.append(instr, qargs, cargs)

    return new_qc


@parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass()
@parametrize_local_mode()
def test_ai_synthesis_keep_original_if_better(
    circuit,
    collector_pass,
    ai_synthesis_pass,
    local_mode,
    test_eagle_backend,
    caplog,
    request,
):
    # FIXME: It looks like when the optimized circuit is worse than the original one, we
    # return a modified version of the original circuit that come from the permutation collection
    if collector_pass == CollectPermutations:
        pytest.skip("Skipping test for permutation until finish FIXME")

    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2),
            ai_synthesis_pass(backend=test_eagle_backend, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert ai_optimized_circuit == original_circuit
    assert all(word in caplog.text for word in ["Running", "synthesis"])
    assert "Keeping the original circuit" in caplog.text


@parametrize_complex_circuit_collector_pass_and_ai_synthesis_pass()
@parametrize_local_mode()
def test_ai_synthesis_pass_with_backend(
    circuit,
    collector_pass,
    ai_synthesis_pass,
    local_mode,
    test_eagle_backend,
    caplog,
    request,
):
    original_circuit = request.getfixturevalue(circuit)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(),
            ai_synthesis_pass(backend=test_eagle_backend, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


# FIXME: Synthesis is not completed for Permutations. Error showed:
# Qargs do not form a connected subgraph of the backend coupling map.
# TODO: Continuing with the previous behavior: on cloud mode we show and
# error and on local mode a warning, decide if we want this or if they should
# have the same behavior
@parametrize_complex_circuit_collector_pass_and_ai_synthesis_pass()
@parametrize_local_mode()
@parametrize_coupling_map_format()
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
    coupling_map = request.getfixturevalue(coupling_map)

    custom_ai_synthesis_pass = PassManager(
        [
            collector_pass(min_block_size=2),
            ai_synthesis_pass(coupling_map=coupling_map, local_mode=local_mode),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(original_circuit)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)
    assert all(word in caplog.text for word in ["Running", "synthesis"])


def test_ai_pass_manager_quantum_volume_example(test_eagle_backend):
    """QuantumVolume test for DAGCircuitError: 'bit mapping invalid:."""

    pass_manager = generate_ai_pass_manager(
        optimization_level=2,
        backend=test_eagle_backend,
        ai_optimization_level=2,
        ai_layout_mode="optimize",
    )

    qv_circuit = quantum_volume(60, seed=42)

    pass_manager.run(qv_circuit)
