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

from qiskit_ibm_transpiler.ai import AIRouting
from qiskit_ibm_transpiler.utils import create_random_linear_function, get_metrics

from .transpiler_service import TranspilerService

logging.basicConfig()
logging.getLogger(__name__).addHandler(logging.NullHandler())
