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

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit.library import (
    IQP,
    ECRGate,
    EfficientSU2,
    ZZFeatureMap,
)
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, random_hermitian
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_ibm_transpiler.transpiler_service import TranspilerService
from qiskit_ibm_transpiler.utils import (
    get_circuit_from_qasm,
    get_circuit_from_qpy,
    get_qpy_from_circuit,
    input_to_qasm,
    to_qasm3_iterative_decomposition,
)
from qiskit_ibm_transpiler.wrappers import _get_circuit_from_result
from tests.parametrize_functions import (
    parametrize_ai,
    parametrize_non_valid_use_fractional_gates,
    parametrize_qiskit_transpile_options,
    parametrize_valid_optimization_level,
    parametrize_valid_optimization_preferences_without_noise,
    parametrize_valid_use_fractional_gates,
)


@parametrize_valid_optimization_level()
@parametrize_ai()
@parametrize_qiskit_transpile_options()
def test_transpiler_service_random_circuit(
    optimization_level, ai, qiskit_transpile_options, test_eagle_backend_name
):
    random_circ = random_circuit(5, depth=3, seed=42)

    cloud_transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai=ai,
        optimization_level=optimization_level,
        qiskit_transpile_options=qiskit_transpile_options,
    )
    transpiled_circuit = cloud_transpiler_service.run(random_circ)

    assert isinstance(transpiled_circuit, QuantumCircuit)


@parametrize_valid_optimization_level()
@parametrize_ai()
@parametrize_qiskit_transpile_options()
def test_transpiler_service_quantum_volume_circuit(
    optimization_level, ai, qiskit_transpile_options, test_eagle_backend_name, qv_circ
):
    cloud_transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai=ai,
        optimization_level=optimization_level,
        qiskit_transpile_options=qiskit_transpile_options,
    )
    transpiled_circuit = cloud_transpiler_service.run(qv_circ)

    assert isinstance(transpiled_circuit, QuantumCircuit)


# FIXME: Code only supports coupling map list format
# @parametrize_coupling_map_format()
@parametrize_valid_optimization_level()
@parametrize_ai()
@parametrize_qiskit_transpile_options()
@parametrize_valid_optimization_preferences_without_noise()
def test_transpiler_service_coupling_map(
    test_eagle_coupling_map_list_format,
    permutation_circuit_test_eagle,
    optimization_level,
    ai,
    qiskit_transpile_options,
    valid_optimization_preferences_without_noise,
):
    # For this tests the circuit is no relevant, so we reuse one we already have
    original_circuit = permutation_circuit_test_eagle

    cloud_transpiler_service = TranspilerService(
        coupling_map=test_eagle_coupling_map_list_format,
        ai=ai,
        optimization_level=optimization_level,
        qiskit_transpile_options=qiskit_transpile_options,
        optimization_preferences=valid_optimization_preferences_without_noise,
    )
    transpiled_circuit = cloud_transpiler_service.run(original_circuit)

    assert isinstance(transpiled_circuit, QuantumCircuit)


@pytest.mark.parametrize("num_circuits", [2, 5])
def test_transpiler_service_several_qv_circuits(
    num_circuits, test_eagle_backend_name, qv_circ
):
    cloud_transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="true",
        optimization_level=1,
    )

    transpiled_circuit = cloud_transpiler_service.run([qv_circ] * num_circuits)
    for circ in transpiled_circuit:
        assert isinstance(circ, QuantumCircuit)


def test_transpiler_service_wrong_input(test_eagle_backend_name, qv_circ):
    cloud_transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="true",
        optimization_level=1,
    )

    circ_dict = {"a": qv_circ}

    with pytest.raises(TypeError):
        cloud_transpiler_service.run(circ_dict)


@parametrize_ai()
def test_transpiler_service_layout_reconstruction(ai):
    n_qubits = 27

    mat = np.real(random_hermitian(n_qubits, seed=1234))
    circuit = IQP(mat)
    observable = SparsePauliOp("Z" * n_qubits)

    cloud_transpiler_service = TranspilerService(
        backend_name="test_eagle",
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


def test_transpiler_service_non_valid_backend_name(
    non_valid_backend_name, basic_cnot_circuit
):
    transpiler_service = TranspilerService(
        backend_name=non_valid_backend_name,
        ai="false",
        optimization_level=3,
    )

    with pytest.raises(
        TranspilerError,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        transpiler_service.run(basic_cnot_circuit)


@pytest.mark.skip(
    "Service accepts now 1e6 gates. Takes too much time to create that circuit."
)
def test_transpiler_service_exceed_circuit_size(test_eagle_backend_name):
    circuit = EfficientSU2(120, entanglement="full", reps=5).decompose()
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="false",
        optimization_level=3,
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert str(e) == "'Circuit has more gates than the allowed maximum of 30000.'"


def test_transpiler_service_exceed_timeout(test_eagle_backend_name):
    circuit = EfficientSU2(100, entanglement="circular", reps=50).decompose()
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="false",
        optimization_level=3,
        timeout=1,
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert (
            "timed out. Try to update the client's timeout config or review your task"
            in str(e)
        )


def test_transpiler_service_wrong_token(test_eagle_backend_name, basic_cnot_circuit):
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="false",
        optimization_level=3,
        token="invented_token5",
    )

    try:
        transpiler_service.run(basic_cnot_circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert str(e) == "'Invalid authentication credentials'"


@pytest.mark.disable_monkeypatch
def test_transpiler_service_wrong_url(test_eagle_backend_name, basic_cnot_circuit):
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="false",
        optimization_level=3,
        base_url="https://ibm.com/",
    )

    with pytest.raises(
        TranspilerError,
        match=r"Internal error: 404 Client Error: Not Found for url: https://www.ibm.com/transpile",
    ):
        transpiler_service.run(basic_cnot_circuit)


@pytest.mark.disable_monkeypatch
def test_transpiler_service_unexisting_url(test_eagle_backend_name, basic_cnot_circuit):
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="false",
        optimization_level=3,
        base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
    )

    with pytest.raises(TranspilerError) as exception_info:
        transpiler_service.run(basic_cnot_circuit)

        assert (
            "Error: HTTPSConnectionPool(host=\\'invented-domain-qiskit-ibm-transpiler-123.com\\', port=443)"
            in exception_info.value
        )


def test_transpiler_service_malformed_body(test_eagle_backend_name, basic_cnot_circuit):
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="false",
        optimization_level=3,
        qiskit_transpile_options={"failing_option": 0},
    )

    with pytest.raises(TranspilerError) as exception_info:
        transpiler_service.run(basic_cnot_circuit)

        assert (
            "\"Error transpiling with Qiskit and qiskit_transpile_options: transpile() got an unexpected keyword argument 'failing_option'\""
            in exception_info.value
        )


def test_transpiler_service_failing_task(
    test_eagle_backend_name, qpy_circuit_with_transpiling_error
):
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        ai="false",
        optimization_level=3,
        coupling_map=[[1, 2], [2, 1]],
        qiskit_transpile_options={
            "basis_gates": ["u1", "u2", "u3", "cx"],
            "seed_transpiler": 0,
        },
    )

    with pytest.raises(TranspilerError) as exception_info:
        transpiler_service.run(get_circuit_from_qpy(qpy_circuit_with_transpiling_error))

        assert "The background task" in exception_info.value
        assert "FAILED" in exception_info.value


def test_transpiler_service_non_valid_circuits_format(
    test_eagle_backend_name, non_valid_qpy_circuit
):
    cloud_transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name, optimization_level=1
    )

    with pytest.raises(TypeError):
        cloud_transpiler_service.run(non_valid_qpy_circuit)


def test_transpiler_service_wrong_qpy_fallback():
    circuit = QuantumCircuit.from_qasm_file("tests/test_files/cc_n64.qasm")
    test_backend = GenericBackendV2(circuit.num_qubits)

    cloud_transpiler_service = TranspilerService(
        coupling_map=list(test_backend.coupling_map.get_edges()), optimization_level=3
    )

    transpiled_circuit = cloud_transpiler_service.run(circuit)

    assert isinstance(transpiled_circuit, QuantumCircuit)


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


def test_transpiler_service_fix_ecr_test_eagle():
    num_qubits = 16
    circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits - 1):
        circuit.ecr(i, i + 1)

    cloud_transpiler_service = TranspilerService(
        backend_name="test_eagle",
        ai="false",
        optimization_level=3,
    )
    transpiled_circuit = cloud_transpiler_service.run(circuit)
    assert any(isinstance(gate.operation, ECRGate) for gate in list(transpiled_circuit))


@parametrize_non_valid_use_fractional_gates()
def test_transpiler_service_non_valid_use_fractional_gates(
    non_valid_use_fractional_gates, test_eagle_backend_name, basic_cnot_circuit
):
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        optimization_level=1,
        use_fractional_gates=non_valid_use_fractional_gates,
    )

    with pytest.raises(TranspilerError) as exception_info:
        transpiler_service.run(basic_cnot_circuit)

        assert "Wrong input" in exception_info.value


@parametrize_valid_use_fractional_gates()
def test_transpiler_service_transpile_valid_use_fractional_gates_param(
    valid_use_fractional_gates, test_eagle_backend_name, basic_cnot_circuit
):
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        optimization_level=1,
        use_fractional_gates=valid_use_fractional_gates,
    )

    transpiled_circuit = transpiler_service.run(basic_cnot_circuit)

    assert isinstance(transpiled_circuit, QuantumCircuit)


def test_transpiler_service_qasm3_iterative_decomposition():
    feature_map = ZZFeatureMap(feature_dimension=3, reps=1, entanglement="full")
    qasm = input_to_qasm(feature_map)
    qc = get_circuit_from_qasm(qasm)
    assert isinstance(qc, QuantumCircuit)


def test_transpiler_service_qasm3_iterative_decomposition_limit():
    feature_map = ZZFeatureMap(feature_dimension=3, reps=1, entanglement="full")
    with pytest.raises(qasm3.QASM3ExporterError):
        to_qasm3_iterative_decomposition(feature_map, n_iter=1)


def test_transpiler_service_barrier_on_circuit(
    test_eagle_backend_name, circuit_with_barrier
):
    transpiler_service = TranspilerService(
        backend_name=test_eagle_backend_name,
        optimization_level=1,
    )

    transpiled_circuit = transpiler_service.run(circuit_with_barrier)

    assert isinstance(transpiled_circuit, QuantumCircuit)
