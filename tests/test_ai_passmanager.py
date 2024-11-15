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

"""Unit-testing generate_ai_pass_manager"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit_ibm_transpiler import generate_ai_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService

BACKEND = QiskitRuntimeService().backend("ibm_brisbane")
COUPLING_MAP = BACKEND.coupling_map


@pytest.mark.parametrize(
    "optimization_level", [1, 2, 3], ids=["opt_level_1", "opt_level_2", "opt_level_3"]
)
@pytest.mark.parametrize(
    "ai_optimization_level",
    [1, 2, 3],
    ids=["ai_opt_level_1", "ai_opt_level_2", "ai_opt_level_3"],
)
@pytest.mark.parametrize(
    "include_ai_synthesis", [False, True], ids=["ai_synthesis", "no_ai_synthesis"]
)
@pytest.mark.parametrize(
    "ai_layout_mode",
    ["keep", "optimize", "improve"],
    ids=["ai_layout_mode_keep", "ai_layout_mode_optimize", "ai_layout_mode_improve"],
)
@pytest.mark.parametrize(
    "qiskit_transpile_options",
    [{}, {"seed_transpiler": 0}],
    ids=["no opt", "one option"],
)
def test_rand_circ_ai_pm(
    optimization_level,
    ai_optimization_level,
    include_ai_synthesis,
    ai_layout_mode,
    qiskit_transpile_options,
):

    su2_circuit = EfficientSU2(10, entanglement="circular", reps=1).decompose()

    ai_transpiler_pass_manager = generate_ai_pass_manager(
        coupling_map=COUPLING_MAP,
        ai_optimization_level=ai_optimization_level,
        include_ai_synthesis=include_ai_synthesis,
        optimization_level=optimization_level,
        ai_layout_mode=ai_layout_mode,
        qiskit_transpile_options=qiskit_transpile_options,
    )
    transpiled_circuit = ai_transpiler_pass_manager.run(su2_circuit)

    assert isinstance(transpiled_circuit, QuantumCircuit)
