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


"""
===============================================================================
Utilities (:mod:`qiskit_ibm_transpiler.utils`)
===============================================================================

.. currentmodule:: qiskit_ibm_transpiler.utils

Functions
=========

.. autofunction:: create_random_linear_function
.. autofunction:: get_metrics

"""

from typing import Dict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.quantum_info import Clifford
from qiskit.synthesis.linear.linear_matrix_utils import random_invertible_binary_matrix


def get_metrics(qc: QuantumCircuit) -> Dict[str, int]:
    """Returns a dict with metrics from a QuantumCircuit"""
    qcd = qc.decompose(reps=3)
    return {
        "n_gates": qcd.size(),
        "n_layers": qcd.depth(),
        "n_cnots": qcd.num_nonlocal_gates(),
        "n_layers_cnots": qcd.depth(lambda x: x[0].name == "cx"),
    }


def random_permutation(n_qubits):
    """Generate a random permutation of n_qubits qubits."""
    return np.random.permutation(n_qubits)


def create_random_linear_function(n_qubits: int, seed: int = 123) -> LinearFunction:
    rand_lin = lambda seed: LinearFunction(
        random_invertible_binary_matrix(n_qubits, seed=seed)
    )

    return LinearFunction(rand_lin(seed))


def random_clifford_from_linear_function(n_qubits: int, seed: int = 123):
    """Generate a random clifford from a random linear function of n_qubits qubits."""

    random_linear = lambda seed: LinearFunction(
        random_invertible_binary_matrix(n_qubits, seed=seed)
    )
    random_clifford = Clifford(random_linear(seed))
    return random_clifford
