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

"""Unit-testing routing_ai"""

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_transpiler_service.ai.routing import AIRouting


@pytest.mark.parametrize("optimization_level", [0, 4, 5])
def test_qv_routing_wrong_opt_level(optimization_level, backend_27q, qv_circ):
    pm = PassManager(
        [AIRouting(optimization_level=optimization_level, backend_name=backend_27q)]
    )
    with pytest.raises(TranspilerError):
        pm.run(qv_circ)


@pytest.mark.parametrize("layout_mode", ["RECREATE", "BOOST"])
def test_qv_routing_wrong_layout_mode(layout_mode, backend_27q, qv_circ):
    with pytest.raises(ValueError):
        PassManager([AIRouting(layout_mode=layout_mode, backend_name=backend_27q)])


def test_routing_wrong_backend(random_circuit_transpiled):
    ai_optimize_lf = PassManager(
        [
            AIRouting(backend_name="wrong_backend"),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
        pytest.fail("Error expected")
    except Exception as e:
        assert (
            "User doesn't have access to the specified backend: wrong_backend" in str(e)
        )


@pytest.mark.skip(
    reason="Unreliable. It passes most of the times with the timeout of 1 second for the current circuits used"
)
def test_routing_exceed_timeout(qv_circ, backend_27q):
    ai_optimize_lf = PassManager(
        [
            AIRouting(backend_name=backend_27q, timeout=1),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(qv_circ)
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_routing_wrong_token(qv_circ, backend_27q):
    ai_optimize_lf = PassManager(
        [
            AIRouting(backend_name=backend_27q, token="invented_token_2"),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(qv_circ)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Invalid authentication credentials" in str(e)


@pytest.mark.disable_monkeypatch
def test_routing_wrong_url(qv_circ, backend_27q):
    ai_optimize_lf = PassManager(
        [
            AIRouting(backend_name=backend_27q, base_url="https://ibm.com/"),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(qv_circ)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Internal error: 404 Client Error: Not Found for url" in str(e)
        assert type(e).__name__ == "TranspilerError"


@pytest.mark.disable_monkeypatch
def test_routing_unexisting_url(qv_circ, backend_27q):
    ai_optimize_lf = PassManager(
        [
            AIRouting(
                backend_name=backend_27q,
                base_url="https://invented-domain-qiskit-transpiler-service-123.com/",
            ),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(qv_circ)
        pytest.fail("Error expected")
    except Exception as e:
        print(e)
        assert (
            "Error: HTTPSConnectionPool(host=\\'invented-domain-qiskit-transpiler-service-123.com\\', port=443):"
            in str(e)
        )
        assert type(e).__name__ == "TranspilerError"


@pytest.mark.parametrize("layout_mode", ["KEEP", "OPTIMIZE", "IMPROVE"])
@pytest.mark.parametrize("optimization_level", [1, 2, 3])
def test_qv_routing(optimization_level, layout_mode, backend_27q, qv_circ):
    pm = PassManager(
        [
            AIRouting(
                optimization_level=optimization_level,
                layout_mode=layout_mode,
                backend_name=backend_27q,
            )
        ]
    )
    circuit = pm.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)
