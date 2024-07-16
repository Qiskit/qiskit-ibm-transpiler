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


@pytest.mark.parametrize("layout_mode", ["KEEP", "OPTIMIZE", "IMPROVE"])
@pytest.mark.parametrize("optimization_level", [1, 2, 3])
def test_qv_routing(optimization_level, layout_mode, backend, qv_circ):
    pm = PassManager(
        [
            AIRouting(
                optimization_level=optimization_level,
                layout_mode=layout_mode,
                backend_name=backend,
            )
        ]
    )
    circuit = pm.run(qv_circ)

    assert isinstance(circuit, QuantumCircuit)


@pytest.mark.parametrize("optimization_level", [0, 4, 5])
def test_qv_routing_wrong_opt_level(optimization_level, backend, qv_circ):
    pm = PassManager(
        [AIRouting(optimization_level=optimization_level, backend_name=backend)]
    )
    with pytest.raises(TranspilerError):
        pm.run(qv_circ)


@pytest.mark.parametrize("layout_mode", ["RECREATE", "BOOST"])
def test_qv_routing_wrong_layout_mode(layout_mode, backend, qv_circ):
    with pytest.raises(ValueError):
        PassManager([AIRouting(layout_mode=layout_mode, backend_name=backend)])
