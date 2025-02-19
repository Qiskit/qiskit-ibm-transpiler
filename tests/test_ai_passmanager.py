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

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2

from qiskit_ibm_transpiler import generate_ai_pass_manager
from tests.parametrize_functions import (
    parametrize_ai_layout_mode,
    parametrize_backend_and_coupling_map,
    parametrize_include_ai_synthesis,
    parametrize_qiskit_transpile_options,
    parametrize_valid_ai_optimization_level,
    parametrize_valid_optimization_level,
)


@parametrize_valid_optimization_level()
@parametrize_valid_ai_optimization_level()
@parametrize_include_ai_synthesis()
@parametrize_ai_layout_mode()
@parametrize_qiskit_transpile_options()
@parametrize_backend_and_coupling_map()
def test_ai_pass_manager(
    optimization_level,
    ai_optimization_level,
    include_ai_synthesis,
    ai_layout_mode,
    qiskit_transpile_options,
    backend_and_coupling_map,
    request,
):

    su2_circuit = EfficientSU2(10, entanglement="circular", reps=1).decompose()

    backend, coupling_map = backend_and_coupling_map
    if coupling_map:
        coupling_map = request.getfixturevalue(coupling_map)

    if backend:
        backend = request.getfixturevalue(backend)

    ai_transpiler_pass_manager = generate_ai_pass_manager(
        backend=backend,
        coupling_map=coupling_map,
        ai_optimization_level=ai_optimization_level,
        include_ai_synthesis=include_ai_synthesis,
        optimization_level=optimization_level,
        ai_layout_mode=ai_layout_mode,
        qiskit_transpile_options=qiskit_transpile_options,
    )
    transpiled_circuit = ai_transpiler_pass_manager.run(su2_circuit)

    assert isinstance(transpiled_circuit, QuantumCircuit)
