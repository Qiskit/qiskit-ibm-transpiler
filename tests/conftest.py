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
import logging

import pytest
from qiskit.transpiler.coupling import CouplingMap


@pytest.fixture(autouse=True)
def env_set(monkeypatch):
    monkeypatch.setenv(
        "QISKIT_TRANSPILER_SERVICE_PERMUTATIONS_URL",
        "https://cloud-transpiler-experimental.quantum-computing.ibm.com/permutations",
    )
    monkeypatch.setenv(
        "QISKIT_TRANSPILER_SERVICE_LINEAR_FUNCTIONS_URL",
        "https://cloud-transpiler-experimental.quantum-computing.ibm.com/linear_functions",
    )
    monkeypatch.setenv(
        "QISKIT_TRANSPILER_SERVICE_CLIFFORD_URL",
        "https://cloud-transpiler-experimental.quantum-computing.ibm.com/clifford",
    )
    monkeypatch.setenv(
        "QISKIT_TRANSPILER_SERVICE_ROUTING_URL",
        "https://cloud-transpiler-experimental.quantum-computing.ibm.com/routing",
    )
    monkeypatch.setenv(
        "QISKIT_TRANSPILER_SERVICE_URL",
        "https://cloud-transpiler-experimental.quantum-computing.ibm.com/",
    )
    logging.getLogger("qiskit_transpiler_service.ai.synthesis").setLevel(logging.DEBUG)


@pytest.fixture(scope="module")
def backend():
    return "ibm_cairo"


@pytest.fixture(scope="module")
def cmap_backend():
    return {
        "ibm_cairo": CouplingMap(
            [
                [0, 1],
                [1, 0],
                [1, 2],
                [1, 4],
                [2, 1],
                [2, 3],
                [3, 2],
                [3, 5],
                [4, 1],
                [4, 7],
                [5, 3],
                [5, 8],
                [6, 7],
                [7, 4],
                [7, 6],
                [7, 10],
                [8, 5],
                [8, 9],
                [8, 11],
                [9, 8],
                [10, 7],
                [10, 12],
                [11, 8],
                [11, 14],
                [12, 10],
                [12, 13],
                [12, 15],
                [13, 12],
                [13, 14],
                [14, 11],
                [14, 13],
                [14, 16],
                [15, 12],
                [15, 18],
                [16, 14],
                [16, 19],
                [17, 18],
                [18, 15],
                [18, 17],
                [18, 21],
                [19, 16],
                [19, 20],
                [19, 22],
                [20, 19],
                [21, 18],
                [21, 23],
                [22, 19],
                [22, 25],
                [23, 21],
                [23, 24],
                [24, 23],
                [24, 25],
                [25, 22],
                [25, 24],
                [25, 26],
                [26, 25],
            ]
        )
    }
