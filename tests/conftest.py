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
from qiskit_ibm_runtime.fake_provider import FakeQuebec


@pytest.fixture(autouse=True)
def env_set(monkeypatch, request):
    if not "disable_monkeypatch" in request.keywords:
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_PERMUTATIONS_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/permutations",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_LINEAR_FUNCTIONS_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/linear_functions",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_CLIFFORD_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/clifford",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_ROUTING_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/routing",
        )
        monkeypatch.setenv(
            "QISKIT_IBM_TRANSPILER_URL",
            "https://cloud-transpiler-experimental.quantum-computing.ibm.com/",
        )
    logging.getLogger("qiskit_ibm_transpiler.ai.synthesis").setLevel(logging.DEBUG)


@pytest.fixture(scope="module")
def backend():
    return "ibm_quebec"


@pytest.fixture(scope="module")
def coupling_map():
    return FakeQuebec().coupling_map


@pytest.fixture(scope="module")
def cmap_backend():
    return {
        "ibm_quebec": CouplingMap(
            [
                [1, 0],
                [2, 1],
                [3, 2],
                [3, 4],
                [4, 15],
                [5, 4],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 16],
                [9, 8],
                [9, 10],
                [11, 10],
                [11, 12],
                [13, 12],
                [14, 0],
                [14, 18],
                [16, 26],
                [17, 12],
                [19, 18],
                [19, 20],
                [20, 33],
                [21, 20],
                [21, 22],
                [22, 15],
                [23, 22],
                [24, 23],
                [25, 24],
                [26, 25],
                [26, 27],
                [28, 27],
                [28, 35],
                [29, 28],
                [30, 17],
                [30, 29],
                [31, 30],
                [31, 32],
                [34, 24],
                [34, 43],
                [35, 47],
                [36, 32],
                [38, 37],
                [38, 39],
                [39, 33],
                [39, 40],
                [40, 41],
                [41, 42],
                [43, 42],
                [43, 44],
                [44, 45],
                [46, 45],
                [47, 46],
                [48, 47],
                [48, 49],
                [50, 49],
                [51, 36],
                [51, 50],
                [52, 37],
                [52, 56],
                [53, 41],
                [53, 60],
                [54, 45],
                [54, 64],
                [55, 49],
                [55, 68],
                [57, 56],
                [58, 57],
                [58, 59],
                [59, 60],
                [60, 61],
                [62, 61],
                [62, 63],
                [62, 72],
                [64, 63],
                [65, 64],
                [65, 66],
                [66, 67],
                [66, 73],
                [68, 67],
                [69, 68],
                [70, 69],
                [70, 74],
                [71, 58],
                [71, 77],
                [74, 89],
                [75, 76],
                [76, 77],
                [77, 78],
                [79, 78],
                [80, 79],
                [80, 81],
                [81, 72],
                [82, 81],
                [82, 83],
                [83, 84],
                [85, 73],
                [85, 84],
                [85, 86],
                [86, 87],
                [87, 88],
                [87, 93],
                [89, 88],
                [90, 75],
                [91, 79],
                [91, 98],
                [92, 83],
                [94, 90],
                [94, 95],
                [96, 95],
                [96, 109],
                [97, 96],
                [97, 98],
                [98, 99],
                [100, 99],
                [100, 110],
                [101, 100],
                [101, 102],
                [102, 92],
                [103, 102],
                [103, 104],
                [104, 105],
                [106, 93],
                [106, 105],
                [106, 107],
                [108, 107],
                [108, 112],
                [109, 114],
                [110, 118],
                [111, 104],
                [114, 113],
                [115, 114],
                [115, 116],
                [116, 117],
                [118, 117],
                [118, 119],
                [120, 119],
                [121, 120],
                [122, 111],
                [122, 121],
                [123, 122],
                [123, 124],
                [124, 125],
                [126, 112],
                [126, 125],
            ]
        )
    }
