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
from qiskit.transpiler import CouplingMap, PassManager

from qiskit_ibm_transpiler.ai.collection import (
    CollectPauliNetworks,
    CollectPermutations,
)
from qiskit_ibm_transpiler.ai.synthesis import AIPermutationSynthesis

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


def test_ai_synthesis_keep_old_circuit():
    orig_qc = QuantumCircuit(127)
    # Add 8qL permutation to find subgraph in current models
    for i, p in enumerate([6, 2, 3, 4, 0, 1, 7, 5]):
        orig_qc.swap(i, p)
    for i, p in enumerate([7, 3, 4, 6, 0, 1, 2, 5]):
        starting_qubit = 37
        orig_qc.swap(i + starting_qubit, p + starting_qubit)
    for i, p in enumerate([5, 0, 4, 2, 6, 3, 1]):
        starting_qubit = 75
        orig_qc.swap(i + starting_qubit, p + starting_qubit)

    cmap_edge_list = [(0, 1), (0, 14), (1, 2), (3, 2), (3, 4), (4, 15), (5, 4), (6, 5), (6, 7), (7, 8), (8, 16), (9, 8), (9, 10), (10, 11), (12, 11), (13, 12), (15, 22), (17, 12), (17, 30), (18, 14), (18, 19), (20, 19), (20, 21), (20, 33), (21, 22), (23, 22), (24, 23), (25, 24), (25, 26), (26, 16), (27, 26), (27, 28), (28, 29), (28, 35), (30, 29), (30, 31), (32, 31), (32, 36), (33, 39), (34, 24), (34, 43), (36, 51), (37, 38), (37, 52), (38, 39), (40, 39), (40, 41), (41, 42), (41, 53), (42, 43), (43, 44), (45, 44), (45, 46), (46, 47), (47, 35), (48, 47), (49, 48), (50, 49), (50, 51), (54, 45), (55, 49), (56, 52), (56, 57), (58, 57), (58, 59), (60, 53), (60, 59), (61, 60), (62, 61), (63, 62), (64, 54), (64, 63), (64, 65), (66, 65), (66, 67), (66, 73), (67, 68), (68, 55), (69, 68), (70, 69), (70, 74), (71, 58), (71, 77), (72, 62), (72, 81), (73, 85), (74, 89), (75, 76), (75, 90), (76, 77), (77, 78), (78, 79), (80, 79), (80, 81), (81, 82), (82, 83), (83, 84), (85, 84), (85, 86), (86, 87), (87, 88), (88, 89), (91, 79), (91, 98), (92, 83), (92, 102), (93, 87), (93, 106), (94, 90), (95, 94), (96, 95), (96, 97), (97, 98), (99, 98), (100, 99), (100, 110), (101, 100), (102, 101), (103, 102), (103, 104), (104, 105), (104, 111), (105, 106), (107, 106), (107, 108), (109, 96), (111, 122), (112, 108), (113, 114), (114, 109), (115, 114), (116, 115), (116, 117), (118, 110), (118, 117), (119, 118), (119, 120), (120, 121), (122, 121), (122, 123), (123, 124), (125, 124), (126, 112), (126, 125)]
    coupling_map = CouplingMap(cmap_edge_list)

    custom_ai_synthesis_pass = PassManager(
        [
            CollectPermutations(min_block_size=2),
            AIPermutationSynthesis(coupling_map=coupling_map, local_mode=True, max_threads=1),
        ]
    )

    ai_optimized_circuit = custom_ai_synthesis_pass.run(orig_qc)

    for inst in ai_optimized_circuit:
        assert inst.operation.name != "permutation"
