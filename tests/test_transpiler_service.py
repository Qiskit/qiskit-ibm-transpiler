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
    EfficientSU2,
    QuantumVolume,
    ECRGate,
    ZZFeatureMap,
)
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp, random_hermitian
from qiskit.providers.fake_provider import GenericBackendV2

from qiskit_ibm_transpiler.transpiler_service import TranspilerService
from qiskit_ibm_transpiler.wrappers import _get_circuit_from_result
from qiskit_ibm_transpiler.utils import (
    get_circuit_from_qasm,
    get_circuit_from_qpy,
    get_qpy_from_circuit,
    input_to_qasm,
    to_qasm3_iterative_decomposition,
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


@pytest.mark.parametrize("num_circuits", [2, 5])
def test_qv_circ_several_circuits_routing(num_circuits):
    qv_circ = QuantumVolume(5, depth=3, seed=42).decompose(reps=3)

    cloud_transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        ai="true",
        optimization_level=1,
    )

    transpiled_circuit = cloud_transpiler_service.run([qv_circ] * num_circuits)
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
            == "\"Error transpiling with Qiskit and qiskit_transpile_options: transpile() got an unexpected keyword argument 'failing_option'\""
        )


def test_transpile_failing_task():
    qpy_circuit = "UUlTS0lUDAECAAAAAAAAAAABZXEAC2YACAAAAAMAAAAAAAAAAAAAAAIAAAABAAAAAAAAAAYAAAAAY2lyY3VpdC0xNjAAAAAAAAAAAHt9cQEAAAADAAEBcQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAABACRnAAAAAgAAAAABAAAAAAAAALsAAAAAAAAAAAAAAAAAAAAAZGN4X2ZkN2JlNTcxOTQ1OTRkZTY5ZDFlMTNmNmFmYmY1NmZjAAtmAAgAAAACAAAAAAAAAAAAAAACAAAAAAAAAAAAAAACAAAAAGNpcmN1aXQtMTYxAAAAAAAAAAB7fQAAAAAAAAAAAAYAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAEAAAABQ1hHYXRlcQAAAABxAAAAAQAGAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAABAAAAAUNYR2F0ZXEAAAABcQAAAAAAAAD///////////////8AAAAAAAAAAAAGAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAABAAAAAUNaR2F0ZXEAAAAAcQAAAAIABwAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABTZGdHYXRlcQAAAAEAJAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABkY3hfZmQ3YmU1NzE5NDU5NGRlNjlkMWUxM2Y2YWZiZjU2ZmNxAAAAAnEAAAABAAYAAAADAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVTNHYXRlcQAAAABmAAAAAAAAAAiMHTg9AR8PQGYAAAAAAAAACH+xa3jilAtAZgAAAAAAAAAItmSFHpiI8j8ABwAAAAEAAAACAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAFDUlhHYXRlcQAAAAFxAAAAAGYAAAAAAAAACI2Sd1JN3gJAAAUAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWUdhdGVxAAAAAgAAAP///////////////wAAAAAAAAAA"
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
        transpiler_service.run(get_circuit_from_qpy(qpy_circuit))
        pytest.fail("Error expected")
    except Exception as e:
        assert "The background task" in str(e)
        assert "FAILED" in str(e)


def test_transpile_wrong_circuits_format():
    circuit = random_circuit(5, depth=3, seed=42).decompose(reps=3)

    cloud_transpiler_service = TranspilerService(
        backend_name="ibm_brisbane", optimization_level=1
    )

    wrong_input = [get_qpy_from_circuit(circuit)] * 2
    with pytest.raises(TypeError):
        cloud_transpiler_service.run(wrong_input)


def test_transpile_wrong_qpy_fallback():
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


def test_layout_construction_no_service(backend_27q, cmap_backend):
    for n_qubits in [5, 10, 15, 20, 27]:
        circuit = random_circuit(n_qubits, 4, measure=True)
        transpile_and_check_layout(cmap_backend[backend_27q], circuit)
    for n_qubits in [5, 10, 15, 20, 27]:
        circuit = EfficientSU2(n_qubits, entanglement="circular", reps=1).decompose()
        transpile_and_check_layout(cmap_backend[backend_27q], circuit)

    for n_qubits in [5, 10, 15, 20, 27]:
        circuit = QuantumCircuit(n_qubits)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.h(4)
        transpile_and_check_layout(cmap_backend[backend_27q], circuit)


def test_fix_ecr_qasm2():
    qc = QuantumCircuit(5)
    qc.ecr(0, 2)

    circuit_from_qasm = get_circuit_from_qasm(qasm2.dumps(qc))
    assert isinstance(list(circuit_from_qasm)[0].operation, ECRGate)


def test_fix_ecr_qasm3():
    qc = QuantumCircuit(5)
    qc.ecr(0, 2)

    circuit_from_qasm = get_circuit_from_qasm(qasm3.dumps(qc))
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


@pytest.mark.parametrize("non_valid_use_fractional_gates_param", [8, "8", "foo"])
def test_transpile_non_valid_use_fractional_gates_param(
    non_valid_use_fractional_gates_param,
):
    circuit = random_circuit(5, depth=3, seed=42)

    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        optimization_level=1,
        use_fractional_gates=non_valid_use_fractional_gates_param,
    )

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Wrong input" in str(e)


@pytest.mark.parametrize(
    "valid_use_fractional_gates_param",
    ["no", "n", "false", "f", "0", "yes", "y", "true", "t", "1"],
)
def test_transpile_valid_use_fractional_gates_param(valid_use_fractional_gates_param):
    circuit = random_circuit(5, depth=3, seed=42)

    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        optimization_level=1,
        use_fractional_gates=valid_use_fractional_gates_param,
    )

    transpiled_circuit = transpiler_service.run(circuit)

    assert isinstance(transpiled_circuit, QuantumCircuit)


def test_qasm3_iterative_decomposition():
    feature_map = ZZFeatureMap(feature_dimension=3, reps=1, entanglement="full")
    qasm = input_to_qasm(feature_map)
    qc = get_circuit_from_qasm(qasm)
    assert isinstance(qc, QuantumCircuit)


def test_qasm3_iterative_decomposition_limit():
    feature_map = ZZFeatureMap(feature_dimension=3, reps=1, entanglement="full")
    with pytest.raises(qasm3.QASM3ExporterError):
        to_qasm3_iterative_decomposition(feature_map, n_iter=1)


def test_transpile_with_barrier_on_circuit():
    circuit = QuantumCircuit(5)
    circuit.x(4)
    circuit.barrier()
    circuit.z(3)
    circuit.cx(3, 4)

    transpiler_service = TranspilerService(
        backend_name="ibm_brisbane",
        optimization_level=1,
    )

    transpiled_circuit = transpiler_service.run(circuit)

    assert isinstance(transpiled_circuit, QuantumCircuit)
