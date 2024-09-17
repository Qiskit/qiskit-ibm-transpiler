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
from qiskit.circuit.library import IQP, EfficientSU2, QuantumVolume, ECRGate
from qiskit.circuit import Gate
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp, random_hermitian

from qiskit_ibm_transpiler.transpiler_service import TranspilerService
from qiskit_ibm_transpiler.wrappers import (
    _get_circuit_from_result,
    _get_circuit_from_qasm,
)


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


@pytest.mark.parametrize(
    "optimization_level", [1, 2, 3], ids=["opt_level_1", "opt_level_2", "opt_level_3"]
)
@pytest.mark.parametrize("ai", ["false", "true"], ids=["no_ai", "ai"])
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
@pytest.mark.parametrize("optimization_level", [1])
@pytest.mark.parametrize("ai", ["false", "true"], ids=["no_ai", "ai"])
@pytest.mark.parametrize("qiskit_transpile_options", [None, {"seed_transpiler": 0}])
@pytest.mark.parametrize("optimization_preferences", [None, "n_cnots"])
def test_rand_circ_cmap_routing(
    coupling_map,
    optimization_level,
    ai,
    qiskit_transpile_options,
    optimization_preferences,
):
    random_circ = random_circuit(5, depth=3, seed=42).decompose(reps=3)

    coupling_map.extend([item[::-1] for item in coupling_map])
    cloud_transpiler_service = TranspilerService(
        coupling_map=coupling_map,
        ai=ai,
        optimization_level=optimization_level,
        qiskit_transpile_options=qiskit_transpile_options,
        optimization_preferences=optimization_preferences,
    )
    transpiled_circuit = cloud_transpiler_service.run(random_circ)

    assert isinstance(transpiled_circuit, QuantumCircuit)


def test_qv_circ_several_circuits_routing():
    qv_circ = QuantumVolume(5, depth=3, seed=42).decompose(reps=3)

    cloud_transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai="true",
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
        ai="true",
        optimization_level=1,
    )

    circ_dict = {"a": qv_circ}
    with pytest.raises(TypeError):
        cloud_transpiler_service.run(circ_dict)


@pytest.mark.parametrize("ai", ["false", "true"], ids=["no_ai", "ai"])
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


@pytest.mark.skip(
    "Service accepts now 1e6 gates. Takes too much time to create that circuit."
)
def test_transpile_exceed_circuit_size():
    circuit = EfficientSU2(120, entanglement="full", reps=5).decompose()
    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai="false",
        optimization_level=3,
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert str(e) == "'Circuit has more gates than the allowed maximum of 30000.'"


def test_transpile_exceed_timeout():
    circuit = EfficientSU2(100, entanglement="circular", reps=50).decompose()
    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
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


def test_transpile_wrong_token():
    circuit = EfficientSU2(100, entanglement="circular", reps=1).decompose()
    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai="false",
        optimization_level=3,
        token="invented_token5",
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert str(e) == "'Invalid authentication credentials'"


@pytest.mark.disable_monkeypatch
def test_transpile_wrong_url():
    circuit = EfficientSU2(100, entanglement="circular", reps=1).decompose()
    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai="false",
        optimization_level=3,
        base_url="https://ibm.com/",
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert (
            "Internal error: 404 Client Error: Not Found for url: https://www.ibm.com/transpile"
            in str(e)
        )
        assert type(e).__name__ == "TranspilerError"


@pytest.mark.disable_monkeypatch
def test_transpile_unexisting_url():
    circuit = EfficientSU2(100, entanglement="circular", reps=1).decompose()
    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai="false",
        optimization_level=3,
        base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert (
            "Error: HTTPSConnectionPool(host=\\'invented-domain-qiskit-ibm-transpiler-123.com\\', port=443)"
            in str(e)
        )


def test_transpile_malformed_body():
    circuit = EfficientSU2(100, entanglement="circular", reps=1).decompose()
    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai="false",
        optimization_level=3,
        qiskit_transpile_options={"failing_option": 0},
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert (
            str(e)
            == "\"transpile() got an unexpected keyword argument 'failing_option'\""
        )


def test_transpile_failing_task():
    open_qasm_circuit = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate dcx q0,q1 { cx q0,q1; cx q1,q0; }\nqreg q[3];\ncz q[0],q[2];\nsdg q[1];\ndcx q[2],q[1];\nu3(3.890139082217223,3.447697582994976,1.1583481971959322) q[0];\ncrx(2.3585459177723522) q[1],q[0];\ny q[2];'
    circuit = QuantumCircuit.from_qasm_str(open_qasm_circuit)
    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
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


def test_fix_ecr_qasm2():
    qc = QuantumCircuit(5)
    qc.ecr(0, 2)

    circuit_from_qasm = _get_circuit_from_qasm(qasm2.dumps(qc))
    assert isinstance(list(circuit_from_qasm)[0].operation, ECRGate)


def test_fix_ecr_qasm3():
    qc = QuantumCircuit(5)
    qc.ecr(0, 2)

    circuit_from_qasm = _get_circuit_from_qasm(qasm3.dumps(qc))
    assert isinstance(list(circuit_from_qasm)[0].operation, ECRGate)


def test_fix_ecr_ibm_strasbourg():
    num_qubits = 16
    circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits - 1):
        circuit.ecr(i, i + 1)

    cloud_transpiler_service = TranspilerService(
        backend_name="ibm_strasbourg",
        ai="false",
        optimization_level=3,
    )
    transpiled_circuit = cloud_transpiler_service.run(circuit)
    assert any(isinstance(gate.operation, ECRGate) for gate in list(transpiled_circuit))
