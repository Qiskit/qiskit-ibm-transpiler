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

"""Tests for the ai routing pass"""

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_ibm_transpiler.ai.routing import AIRouting
from tests.parametrize_functions import (
    parametrize_coupling_map_format,
    parametrize_local_mode,
    parametrize_local_mode_and_error_type,
    parametrize_non_valid_layout_mode,
    parametrize_non_valid_optimization_level,
    parametrize_non_valid_optimization_preferences,
    parametrize_valid_layout_mode,
    parametrize_valid_optimization_level,
    parametrize_valid_optimization_preferences,
)


@pytest.mark.skip(
    reason="Unreliable. It passes most of the times with the timeout of 1 second for the current circuits used"
)
def test_ai_cloud_routing_pass_exceed_timeout(qv_circ, test_eagle_backend_name):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                backend_name=test_eagle_backend_name, timeout=1, local_mode=False
            ),
        ]
    )

    ai_optimized_circuit = ai_routing_pass.run(qv_circ)

    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_ai_cloud_routing_pass_wrong_token(qv_circ, test_eagle_backend_name):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                backend_name=test_eagle_backend_name,
                token="invented_token_2",
                local_mode=False,
            ),
        ]
    )

    try:
        ai_routing_pass.run(qv_circ)
        pytest.fail("Error expected")
    except Exception as e:
        assert "Invalid authentication credentials" in str(e)


@pytest.mark.disable_monkeypatch
def test_ai_cloud_routing_pass_wrong_url(qv_circ, test_eagle_backend_name):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                backend_name=test_eagle_backend_name,
                base_url="https://ibm.com/",
                local_mode=False,
            ),
        ]
    )

    with pytest.raises(TranspilerError) as exception_info:
        ai_routing_pass.run(qv_circ)

        assert (
            "Internal error: 404 Client Error: Not Found for url"
            in exception_info.value
        )


@pytest.mark.disable_monkeypatch
def test_ai_cloud_routing_pass_unexisting_url(qv_circ, test_eagle_backend_name):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                backend_name=test_eagle_backend_name,
                base_url="https://invented-domain-qiskit-ibm-transpiler-123.com/",
                local_mode=False,
            ),
        ]
    )

    with pytest.raises(TranspilerError) as exception_info:
        ai_routing_pass.run(qv_circ)

        assert (
            "Error: HTTPSConnectionPool(host=\\'invented-domain-qiskit-ibm-transpiler-123.com\\', port=443):"
            in exception_info.value
        )


@parametrize_local_mode_and_error_type()
def test_ai_routing_pass_wrong_backend(error_type, local_mode, basic_cnot_circuit):
    with pytest.raises(
        error_type,
        match=r"User doesn\'t have access to the specified backend: \w+",
    ):
        ai_routing_pass = PassManager(
            [
                AIRouting(backend_name="wrong_backend", local_mode=local_mode),
            ]
        )
        ai_routing_pass.run(basic_cnot_circuit)


@parametrize_non_valid_optimization_level()
@parametrize_local_mode()
def test_ai_routing_pass_non_valid_optimization_level(
    optimization_level, local_mode, test_eagle_backend_name
):
    with pytest.raises(
        ValueError,
        match=r"ERROR. The optimization_level should be a value between 1 and 3.",
    ):

        PassManager(
            [
                AIRouting(
                    optimization_level=optimization_level,
                    backend_name=test_eagle_backend_name,
                    local_mode=local_mode,
                )
            ]
        )


@parametrize_valid_optimization_level()
@parametrize_local_mode()
def test_ai_routing_pass_valid_optimization_level(
    optimization_level,
    local_mode,
    test_eagle_backend_name,
    qv_circ,
):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                optimization_level=optimization_level,
                backend_name=test_eagle_backend_name,
                local_mode=local_mode,
            )
        ]
    )

    circuit = ai_routing_pass.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)


@parametrize_non_valid_optimization_preferences()
@parametrize_local_mode()
def test_ai_routing_pass_non_valid_optimization_preferences(
    non_valid_optimization_preferences, local_mode, test_eagle_backend_name
):
    with pytest.raises(
        ValueError,
        match=r"'\w+' is not a valid optimization preference",
    ):

        PassManager(
            [
                AIRouting(
                    optimization_preferences=non_valid_optimization_preferences,
                    backend_name=test_eagle_backend_name,
                    local_mode=local_mode,
                )
            ]
        )


@parametrize_valid_optimization_preferences()
@parametrize_local_mode()
def test_ai_routing_pass_valid_optimization_preferences(
    valid_optimization_preferences,
    local_mode,
    test_eagle_backend_name,
    qv_circ,
):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                backend_name=test_eagle_backend_name,
                optimization_preferences=valid_optimization_preferences,
                local_mode=local_mode,
            )
        ]
    )

    circuit = ai_routing_pass.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)


@parametrize_non_valid_layout_mode()
@parametrize_local_mode()
def test_ai_routing_pass_non_valid_layout_mode(
    layout_mode, local_mode, test_eagle_backend_name
):
    with pytest.raises(ValueError):
        PassManager(
            [
                AIRouting(
                    layout_mode=layout_mode,
                    backend_name=test_eagle_backend_name,
                    local_mode=local_mode,
                )
            ]
        )


@parametrize_valid_layout_mode()
@parametrize_local_mode()
def test_ai_routing_pass_valid_layout_mode(
    layout_mode,
    local_mode,
    test_eagle_backend_name,
    qv_circ,
):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                layout_mode=layout_mode,
                backend_name=test_eagle_backend_name,
                local_mode=local_mode,
            )
        ]
    )

    circuit = ai_routing_pass.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)


@parametrize_local_mode()
def test_ai_routing_pass_with_backend_name(
    local_mode,
    test_eagle_backend_name,
    qv_circ,
):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                backend_name=test_eagle_backend_name,
                local_mode=local_mode,
            )
        ]
    )

    circuit = ai_routing_pass.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)


@parametrize_local_mode()
def test_ai_routing_pass_with_backend(
    local_mode,
    test_eagle_backend,
    qv_circ,
):
    ai_routing_pass = PassManager(
        [
            AIRouting(
                backend=test_eagle_backend,
                local_mode=local_mode,
            )
        ]
    )

    circuit = ai_routing_pass.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)


@parametrize_coupling_map_format()
@parametrize_local_mode()
def test_ai_routing_pass_with_coupling_map(
    coupling_map,
    local_mode,
    qv_circ,
    request,
):
    coupling_map = request.getfixturevalue(coupling_map)

    ai_routing_pass = PassManager(
        [
            AIRouting(
                coupling_map=coupling_map,
                local_mode=local_mode,
            )
        ]
    )

    circuit = ai_routing_pass.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)
