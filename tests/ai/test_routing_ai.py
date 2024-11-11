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

from qiskit_ibm_transpiler.ai.routing import AIRouting


@pytest.mark.parametrize("optimization_level", [0, 4, 5])
def test_qv_routing_wrong_opt_level(optimization_level, backend, qv_circ):
    pm = PassManager(
        [
            AIRouting(
                optimization_level=optimization_level,
                backend_name=backend,
                local_mode=False,
            )
        ]
    )
    with pytest.raises(TranspilerError):
        pm.run(qv_circ)


@pytest.mark.parametrize("optimization_preferences", ["foo"])
def test_qv_routing_wrong_opt_preferences(optimization_preferences, backend, qv_circ):
    pm = PassManager(
        [
            AIRouting(
                optimization_preferences=optimization_preferences,
                backend_name=backend,
                local_mode=False,
            )
        ]
    )
    with pytest.raises(TranspilerError):
        pm.run(qv_circ)


@pytest.mark.parametrize("layout_mode", ["RECREATE", "BOOST"])
def test_qv_routing_wrong_layout_mode(layout_mode, backend, qv_circ):
    with pytest.raises(ValueError):
        PassManager(
            [AIRouting(layout_mode=layout_mode, backend_name=backend, local_mode=False)]
        )


def test_routing_wrong_backend(random_circuit_transpiled):
    with pytest.raises(
        TranspilerError,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        ai_optimize_lf = PassManager(
            [
                AIRouting(backend_name="wrong_backend", local_mode=False),
            ]
        )
        ai_optimize_lf.run(random_circuit_transpiled)


@pytest.mark.skip(
    reason="Unreliable. It passes most of the times with the timeout of 1 second for the current circuits used"
)
def test_routing_exceed_timeout(qv_circ, backend):
    ai_optimize_lf = PassManager(
        [
            AIRouting(backend_name=backend, timeout=1, local_mode=False),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(qv_circ)
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_routing_wrong_token(qv_circ, backend):
    ai_optimize_lf = PassManager(
        [
            AIRouting(backend_name=backend, token="invented_token_2", local_mode=False),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(qv_circ)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Invalid authentication credentials" in str(e)


@pytest.mark.disable_monkeypatch
def test_routing_wrong_url(qv_circ, backend):
    ai_optimize_lf = PassManager(
        [
            AIRouting(
                backend_name=backend, base_url="https://ibm.com/", local_mode=False
            ),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(qv_circ)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Internal error: 404 Client Error: Not Found for url" in str(e)
        assert type(e).__name__ == "TranspilerError"


@pytest.mark.disable_monkeypatch
def test_routing_unexisting_url(qv_circ, backend):
    ai_optimize_lf = PassManager(
        [
            AIRouting(
                backend_name=backend,
                base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
                local_mode=False,
            ),
        ]
    )
    try:
        ai_optimized_circuit = ai_optimize_lf.run(qv_circ)
        pytest.fail("Error expected")
    except Exception as e:
        print(e)
        assert (
            "Error: HTTPSConnectionPool(host=\\'invented-domain-qiskit-ibm-transpiler-123.com\\', port=443):"
            in str(e)
        )
        assert type(e).__name__ == "TranspilerError"


@pytest.mark.parametrize("layout_mode", ["KEEP", "OPTIMIZE", "IMPROVE"])
@pytest.mark.parametrize("optimization_level", [1, 2, 3])
@pytest.mark.parametrize(
    "optimization_preferences", [None, "noise", ["noise", "n_cnots"]]
)
def test_qv_routing(
    optimization_level, layout_mode, optimization_preferences, backend, qv_circ
):
    pm = PassManager(
        [
            AIRouting(
                optimization_level=optimization_level,
                layout_mode=layout_mode,
                backend_name=backend,
                optimization_preferences=optimization_preferences,
                local_mode=False,
            )
        ]
    )
    circuit = pm.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)
