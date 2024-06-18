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

"""Unit-testing Transpiler Service"""

import logging

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit.library import IQP, EfficientSU2, QuantumVolume
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp, random_hermitian

from qiskit_transpiler_service.transpiler_service import TranspilerService
from qiskit_transpiler_service.wrappers import _get_circuit_from_result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.mark.parametrize(
    "optimization_level", [1, 2, 3], ids=["opt_level_1", "opt_level_2", "opt_level_3"]
)
@pytest.mark.parametrize("ai", [False, True], ids=["no_ai", "ai"])
@pytest.mark.parametrize(
    "qiskit_transpile_options",
    [None, {"seed_transpiler": 0}],
    ids=["no opt", "one option"],
)
def test_rand_circ_backend_routing(optimization_level, ai, qiskit_transpile_options):
    backend_name = "ibm_brisbane"
    random_circ = random_circuit(5, depth=3, seed=42)

    cloud_transpiler_service = TranspilerService(
        backend_name=backend_name,
        ai=ai,
        optimization_level=optimization_level,
        qiskit_transpile_options=qiskit_transpile_options,
    )
    transpiled_circuit = cloud_transpiler_service.run(random_circ)

    assert isinstance(transpiled_circuit, QuantumCircuit)


@pytest.mark.parametrize(
    "optimization_level", [1, 2, 3], ids=["opt_level_1", "opt_level_2", "opt_level_3"]
)
@pytest.mark.parametrize("ai", [False, True], ids=["no_ai", "ai"])
@pytest.mark.parametrize(
    "qiskit_transpile_options",
    [None, {"seed_transpiler": 0}],
    ids=["no opt", "one option"],
)
def test_qv_backend_routing(optimization_level, ai, qiskit_transpile_options):
    backend_name = "ibm_brisbane"
    qv_circ = QuantumVolume(27, depth=3, seed=42).decompose(reps=3)

    cloud_transpiler_service = TranspilerService(
        backend_name=backend_name,
        ai=ai,
        optimization_level=optimization_level,
        qiskit_transpile_options=qiskit_transpile_options,
    )
    transpiled_circuit = cloud_transpiler_service.run(qv_circ)

    assert isinstance(transpiled_circuit, QuantumCircuit)


@pytest.mark.parametrize(
    "coupling_map",
    [
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
    ],
)
@pytest.mark.parametrize("optimization_level", [1, 2, 3])
@pytest.mark.parametrize("ai", [False, True], ids=["no_ai", "ai"])
@pytest.mark.parametrize("qiskit_transpile_options", [None, {"seed_transpiler": 0}])
def test_rand_circ_cmap_routing(
    coupling_map, optimization_level, ai, qiskit_transpile_options
):
    random_circ = random_circuit(5, depth=3, seed=42).decompose(reps=3)

    cloud_transpiler_service = TranspilerService(
        coupling_map=coupling_map,
        ai=ai,
        optimization_level=optimization_level,
        qiskit_transpile_options=qiskit_transpile_options,
    )
    transpiled_circuit = cloud_transpiler_service.run(random_circ)

    assert isinstance(transpiled_circuit, QuantumCircuit)


def test_qv_circ_several_circuits_routing():
    qv_circ = QuantumVolume(5, depth=3, seed=42).decompose(reps=3)

    cloud_transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai=True,
        optimization_level=1,
    )
    transpiled_circuit = cloud_transpiler_service.run([qv_circ] * 2)
    for circ in transpiled_circuit:
        assert isinstance(circ, QuantumCircuit)

    transpiled_circuit = cloud_transpiler_service.run([qasm2.dumps(qv_circ)] * 2)
    for circ in transpiled_circuit:
        assert isinstance(circ, QuantumCircuit)

    transpiled_circuit = cloud_transpiler_service.run([qasm2.dumps(qv_circ), qv_circ])
    for circ in transpiled_circuit:
        assert isinstance(circ, QuantumCircuit)


def test_qv_circ_wrong_input_routing():
    qv_circ = QuantumVolume(5, depth=3, seed=42).decompose(reps=3)

    cloud_transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai=True,
        optimization_level=1,
    )

    circ_dict = {"a": qv_circ}
    with pytest.raises(TypeError):
        cloud_transpiler_service.run(circ_dict)


@pytest.mark.parametrize("ai", [False, True], ids=["no_ai", "ai"])
def test_transpile_layout_reconstruction(ai):
    n_qubits = 27

    mat = np.real(random_hermitian(n_qubits, seed=1234))
    circuit = IQP(mat)
    observable = SparsePauliOp("Z" * n_qubits)

    cloud_transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai=ai,
        optimization_level=1,
    )
    transpiled_circuit = cloud_transpiler_service.run(circuit)
    # This fails if initial layout is not correct
    try:
        observable.apply_layout(transpiled_circuit.layout)
    except Exception:
        pytest.fail(
            "This should not fail. Probably something wrong with the reconstructed layout."
        )


def compare_layouts(plugin_circ, non_ai_circ):
    assert (
        plugin_circ.layout.initial_layout == non_ai_circ.layout.initial_layout
    ), "initial_layouts differs"
    assert (
        plugin_circ.layout.initial_index_layout()
        == non_ai_circ.layout.initial_index_layout()
    ), "initial_index_layout differs"
    assert (
        plugin_circ.layout.input_qubit_mapping == non_ai_circ.layout.input_qubit_mapping
    ), "input_qubit_mapping differs"
    assert (
        plugin_circ.layout._input_qubit_count == non_ai_circ.layout._input_qubit_count
    ), "_input_qubit_count differs"
    assert (
        plugin_circ.layout._output_qubit_list == non_ai_circ.layout._output_qubit_list
    ), "_output_qubit_list differs"
    # Sometimes qiskit transpilation does not add final_layout
    if non_ai_circ.layout.final_layout:
        assert (
            plugin_circ.layout.final_layout == non_ai_circ.layout.final_layout
        ), "final_layout differs"
    assert (
        plugin_circ.layout.final_index_layout()
        == non_ai_circ.layout.final_index_layout()
    ), "final_index_layout differs"


def get_circuit_as_in_service(circuit):
    return {
        "qasm": qasm3.dumps(circuit),
        "layout": {
            "initial": circuit.layout.initial_index_layout(),
            "final": circuit.layout.final_index_layout(False),
        },
    }


def transpile_and_check_layout(cmap, circuit):
    non_ai_circ = transpile(
        circuits=circuit,
        coupling_map=cmap,
        optimization_level=1,
    )
    service_resp = get_circuit_as_in_service(non_ai_circ)
    plugin_circ = _get_circuit_from_result(service_resp, circuit)
    compare_layouts(plugin_circ, non_ai_circ)


def test_layout_construction_no_service(backend, cmap_backend):
    for n_qubits in [5, 10, 15, 20, 27]:
        circuit = random_circuit(n_qubits, 4, measure=True)
        transpile_and_check_layout(cmap_backend[backend], circuit)
    for n_qubits in [5, 10, 15, 20, 27]:
        circuit = EfficientSU2(n_qubits, entanglement="circular", reps=1).decompose()
        transpile_and_check_layout(cmap_backend[backend], circuit)

    for n_qubits in [5, 10, 15, 20, 27]:
        circuit = QuantumCircuit(n_qubits)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.h(4)
        transpile_and_check_layout(cmap_backend[backend], circuit)
