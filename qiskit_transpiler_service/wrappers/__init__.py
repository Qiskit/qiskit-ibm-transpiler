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
AI (:mod:`qiskit_ibm_transpiler.wrappers`)
===============================================================================

.. currentmodule:: qiskit_ibm_transpiler.wrappers

Classes
=======
.. autosummary::
   :toctree: ../stubs/

   AIRoutingAPI
   AICliffordAPI
   AILinearFunctionAPI
   AIPermutationAPI
   QiskitTranspilerService
   BackendTaskError
   TranspileAPI
"""

from qiskit_ibm_transpiler.wrappers.ai_routing import AIRoutingAPI
from qiskit_ibm_transpiler.wrappers.ai_synthesis import (
    AICliffordAPI,
    AILinearFunctionAPI,
    AIPermutationAPI,
)
from qiskit_ibm_transpiler.wrappers.base import (
    BackendTaskError,
    QiskitTranspilerService,
)
from qiskit_ibm_transpiler.wrappers.transpile import (
    TranspileAPI,
    _get_circuit_from_result,
)
