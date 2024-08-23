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

import pytest
from warnings import catch_warnings
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit.library import IQP, EfficientSU2, QuantumVolume
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile

from qiskit_transpiler_service.transpiler_service import TranspilerService
from qiskit_transpiler_service.wrappers import _get_circuit_from_result


@pytest.mark.parametrize(
    "optimization_level", [1, 2, 3], ids=["opt_level_1", "opt_level_2", "opt_level_3"]
)
@pytest.mark.parametrize("ai", ["false", "true"], ids=["no_ai", "ai"])
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


def test_qv_circ_wrong_input_routing():
    qv_circ = QuantumVolume(5, depth=3, seed=42).decompose(reps=3)

    cloud_transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai="true",
        optimization_level=1,
    )

    circ_dict = {"a": qv_circ}
    with pytest.raises(TypeError):
        cloud_transpiler_service.run(circ_dict)


def test_transpile_non_valid_backend():
    circuit = EfficientSU2(100, entanglement="circular", reps=1).decompose()
    non_valid_backend_name = "ibm_torin"
    transpiler_service = TranspilerService(
        backend_name=non_valid_backend_name,
        ai="false",
        optimization_level=3,
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert (
            str(e)
            == f'"User doesn\'t have access to the specified backend: {non_valid_backend_name}"'
        )


def test_transpile_wrong_token():
    circuit = EfficientSU2(100, entanglement="circular", reps=1).decompose()
    transpiler_service = TranspilerService(
        backend_name="ibm_kyoto",
        ai="false",
        optimization_level=3,
        token="invented_token5",
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert str(e) == "'Invalid authentication credentials'"


def test_transpile_failing_task():
    open_qasm_circuit = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate dcx q0,q1 { cx q0,q1; cx q1,q0; }\nqreg q[3];\ncz q[0],q[2];\nsdg q[1];\ndcx q[2],q[1];\nu3(3.890139082217223,3.447697582994976,1.1583481971959322) q[0];\ncrx(2.3585459177723522) q[1],q[0];\ny q[2];'
    circuit = QuantumCircuit.from_qasm_str(open_qasm_circuit)
    transpiler_service = TranspilerService(
        backend_name="ibm_kyoto",
        ai="false",
        optimization_level=3,
        coupling_map=[[1, 2], [2, 1]],
        qiskit_transpile_options={
            "basis_gates": ["u1", "u2", "u3", "cx"],
            "seed_transpiler": 0,
        },
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert "The background task" in str(e)
        assert "FAILED" in str(e)


def test_deprecation_warning():
    with catch_warnings(record=True) as w:
        TranspilerService(
            backend_name="ibm_brisbane",
            ai="true",
            optimization_level=1,
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert (
            str(w[0].message)
            == "The package qiskit_transpiler_service is deprecated. Use qiskit_ibm_transpiler instead"
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
