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

"""Unit-testing permutation_ai"""
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_transpiler_service.ai.collection import CollectPermutations
from qiskit_transpiler_service.ai.synthesis import AIPermutationSynthesis


@pytest.fixture
def permutations_circuit(backend, cmap_backend):
    coupling_map = cmap_backend[backend]
    cmap = list(coupling_map.get_edges())
    orig_qc = QuantumCircuit(27)
    for i, j in cmap:
        orig_qc.h(i)
        orig_qc.cx(i, j)
    for i, j in cmap:
        orig_qc.swap(i, j)
    for i, j in cmap:
        orig_qc.h(i)
        orig_qc.cx(i, j)
    for i, j in cmap[:4]:
        orig_qc.swap(i, j)
    for i, j in cmap:
        orig_qc.cx(i, j)
    return orig_qc


def test_permutation_collector(permutations_circuit, backend, cmap_backend):
    qiskit_lvl3_transpiler = generate_preset_pass_manager(
        optimization_level=1, coupling_map=cmap_backend[backend]
    )
    permutations_circuit = qiskit_lvl3_transpiler.run(permutations_circuit)

    pm = PassManager(
        [
            CollectPermutations(max_block_size=27),
        ]
    )
    perm_only_circ = pm.run(permutations_circuit)
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(perm_only_circ)
    perm_nodes = dag.named_nodes("permutation", "Permutation")
    assert len(perm_nodes) == 2
    assert perm_nodes[0].op.num_qubits == 27
    assert perm_nodes[1].op.num_qubits == 4
    assert not dag.named_nodes("linear_function", "Linear_function")
    assert not dag.named_nodes("clifford", "Clifford")


def test_permutation_pass(permutations_circuit, backend, cmap_backend, caplog):
    qiskit_lvl3_transpiler = generate_preset_pass_manager(
        optimization_level=1, coupling_map=cmap_backend[backend]
    )
    permutations_circuit = qiskit_lvl3_transpiler.run(permutations_circuit)

    ai_optimize_lf = PassManager(
        [
            CollectPermutations(max_block_size=27),
            AIPermutationSynthesis(backend_name=backend),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(permutations_circuit)
    assert "Using the synthesized circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_permutation_wrong_backend(caplog):
    orig_qc = QuantumCircuit(3)
    orig_qc.swap(0, 1)
    orig_qc.swap(1, 2)

    ai_optimize_lf = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(backend_name="wrong_backend"),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(orig_qc)
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
def test_permutation_exceed_timeout(random_circuit_transpiled, backend, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(backend_name=backend, timeout=1),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_permutation_wrong_token(random_circuit_transpiled, backend, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(backend_name=backend, token="invented_token_2"),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert "Invalid authentication credentials" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_permutation_wrong_url(monkeypatch, random_circuit_transpiled, backend):
    monkeypatch.undo()
    ai_optimize_lf = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(
                backend_name=backend, base_url="https://ibm.com/"
            ),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Expecting value: line 1 column 1 (char 0)" in str(
            e
        )
        assert type(e).__name__ == "JSONDecodeError"


def test_permutation_unexisting_url(
    monkeypatch, random_circuit_transpiled, backend, caplog
):
    monkeypatch.undo()
    ai_optimize_lf = PassManager(
        [
            CollectPermutations(min_block_size=2, max_block_size=27),
            AIPermutationSynthesis(
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
