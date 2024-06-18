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

"""Unit-testing clifford_ai"""
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_transpiler_service.ai.collection import CollectCliffords
from qiskit_transpiler_service.ai.synthesis import AICliffordSynthesis


def test_clifford_function(random_circuit_transpiled, backend):
    ai_optimize_lf = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(backend_name=backend),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert isinstance(ai_optimized_circuit, QuantumCircuit)


def test_clifford_wrong_backend(random_circuit_transpiled, caplog):
    ai_optimize_lf = PassManager(
        [
            CollectCliffords(),
            AICliffordSynthesis(backend_name="wrong_backend"),
        ]
    )
    ai_optimized_circuit = ai_optimize_lf.run(random_circuit_transpiled)
    assert "couldn't synthesize the circuit" in caplog.text
    assert "Keeping the original circuit" in caplog.text
    assert isinstance(ai_optimized_circuit, QuantumCircuit)
