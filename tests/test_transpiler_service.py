# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit-testing Transpiler Service"""

import pytest
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit.library import (
    ECRGate,
    EfficientSU2,
    ZZFeatureMap,
)
from qiskit.compiler import transpile

from qiskit_ibm_transpiler.transpiler_service import TranspilerService
from qiskit_ibm_transpiler.utils import (
    get_circuit_from_qasm,
    get_qpy_from_circuit,
    input_to_qasm,
    to_qasm3_iterative_decomposition,
)
from qiskit_ibm_transpiler.wrappers import _get_circuit_from_result


@pytest.mark.parametrize("ai", ["true", "false", "auto"])
def test_transpiler_service_function_deprecated(ai):
    with pytest.raises(ValueError, match="ibm/transpiler-function.*deprecated"):
        TranspilerService(optimization_level=1, ai=ai)


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
        "qpy": get_qpy_from_circuit(circuit),
        "qasm": None,
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


def test_transpiler_service_layout_construction_no_service(test_eagle_coupling_map):
    # FIXME: Test fail when uncommenting this code. The error msg is different each run.
    # for n_qubits in [5, 30, 60, 90, 120, 127]:
    #     circuit = random_circuit(n_qubits, 4, measure=True)
    #     transpile_and_check_layout(test_eagle_coupling_map, circuit)

    for n_qubits in [5, 30, 60, 90, 120, 127]:
        circuit = EfficientSU2(n_qubits, entanglement="circular", reps=1).decompose()
        transpile_and_check_layout(test_eagle_coupling_map, circuit)

    for n_qubits in [5, 30, 60, 90, 120, 127]:
        circuit = QuantumCircuit(n_qubits)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.h(4)
        transpile_and_check_layout(test_eagle_coupling_map, circuit)


def test_transpiler_service_fix_ecr_qasm2():
    qc = QuantumCircuit(5)
    qc.ecr(0, 2)

    circuit_from_qasm = get_circuit_from_qasm(qasm2.dumps(qc))
    assert isinstance(list(circuit_from_qasm)[0].operation, ECRGate)


def test_transpiler_service_fix_ecr_qasm3():
    qc = QuantumCircuit(5)
    qc.ecr(0, 2)

    circuit_from_qasm = get_circuit_from_qasm(qasm3.dumps(qc))
    assert isinstance(list(circuit_from_qasm)[0].operation, ECRGate)


def test_transpiler_service_qasm3_iterative_decomposition():
    feature_map = ZZFeatureMap(feature_dimension=3, reps=1, entanglement="full")
    qasm = input_to_qasm(feature_map)
    qc = get_circuit_from_qasm(qasm)
    assert isinstance(qc, QuantumCircuit)


def test_transpiler_service_qasm3_iterative_decomposition_limit():
    feature_map = ZZFeatureMap(feature_dimension=3, reps=1, entanglement="full")
    with pytest.raises(qasm3.QASM3ExporterError):
        to_qasm3_iterative_decomposition(feature_map, n_iter=1)
