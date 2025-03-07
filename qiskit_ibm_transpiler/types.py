# -*- coding: utf-8 -*-

# (C) Copyright 2025 IBM. All Rights Reserved.
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
Qiskit IBM Transpiler - Type Definitions (:mod:`qiskit_ibm_transpiler.types`)
===============================================================================

This module defines shared type aliases used throughout the `qiskit_ibm_transpiler`
package, centralizing type definitions.

.. currentmodule:: qiskit_ibm_transpiler.types

Type Aliases
============
This module provides common type definitions used across multiple transpiler components.

- **OptimizationOptions**: Specifies different optimization strategies for quantum circuits.

Classes
=======
.. autosummary::
   :toctree: ../stubs/

   TranspilerService
"""

from typing import Literal

OptimizationOptions = Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]
