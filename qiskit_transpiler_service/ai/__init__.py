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
AI (:mod:`qiskit_transpiler_service.ai`)
===============================================================================

.. currentmodule:: qiskit_transpiler_service.ai

Classes
=======
.. autosummary::
   :toctree: ../stubs/

   AIRouting
   AICliffordSynthesis
   AILinearFunctionSynthesis
   AIPermutationSynthesis
   CollectCliffords
   CollectLinearFunctions
   CollectPermutations
"""
from .collection import (
    CollectCliffords,  # noqa: F401
    CollectLinearFunctions,  # noqa: F401
    CollectPermutations,  # noqa: F401
)
from .routing import AIRouting  # noqa: F401
from .synthesis import (
    AICliffordSynthesis,  # noqa: F401
    AILinearFunctionSynthesis,  # noqa: F401
    AIPermutationSynthesis,  # noqa: F401
)
