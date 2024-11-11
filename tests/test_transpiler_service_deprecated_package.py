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

from qiskit_transpiler_service.transpiler_service import TranspilerService


@pytest.mark.parametrize("ai", ["false", "true"], ids=["no_ai", "ai"])
def test_rand_circ_backend_routing(ai):
    backend_name = "ibm_brisbane"
    random_circ = random_circuit(5, depth=3, seed=42)

    with catch_warnings(record=True) as w:
        cloud_transpiler_service = TranspilerService(
            backend_name=backend_name,
            ai=ai,
            optimization_level=1,
            qiskit_transpile_options=None,
        )
        assert_deprecation_warning(w)

    transpiled_circuit = cloud_transpiler_service.run(random_circ)

    assert isinstance(transpiled_circuit, QuantumCircuit)


def test_qv_circ_wrong_input_routing():
    qv_circ = QuantumVolume(5, depth=3, seed=42).decompose(reps=3)

    with catch_warnings(record=True) as w:
        cloud_transpiler_service = TranspilerService(
            backend_name="ibm_brisbane",
            ai="true",
            optimization_level=1,
        )
        assert_deprecation_warning(w)

    circ_dict = {"a": qv_circ}
    with pytest.raises(TypeError):
        cloud_transpiler_service.run(circ_dict)


def test_transpile_wrong_token():
    circuit = EfficientSU2(100, entanglement="circular", reps=1).decompose()
    with catch_warnings(record=True) as w:
        transpiler_service = TranspilerService(
            backend_name="ibm_brisbane",
            ai="false",
            optimization_level=3,
            token="invented_token5",
        )
        assert_deprecation_warning(w)

    try:
        transpiler_service.run(circuit)
        pytest.fail("Error expected")
    except Exception as e:
        assert str(e) == "'Invalid authentication credentials'"


def test_transpile_failing_task():
    open_qasm_circuit = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate dcx q0,q1 { cx q0,q1; cx q1,q0; }\nqreg q[3];\ncz q[0],q[2];\nsdg q[1];\ndcx q[2],q[1];\nu3(3.890139082217223,3.447697582994976,1.1583481971959322) q[0];\ncrx(2.3585459177723522) q[1],q[0];\ny q[2];'
    circuit = QuantumCircuit.from_qasm_str(open_qasm_circuit)
    with catch_warnings(record=True) as w:
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
        assert_deprecation_warning(w)

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
        assert_deprecation_warning(w)


def assert_deprecation_warning(w):
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert (
        str(w[0].message)
        == "The package qiskit_transpiler_service is deprecated. Use qiskit_ibm_transpiler instead"
    )
