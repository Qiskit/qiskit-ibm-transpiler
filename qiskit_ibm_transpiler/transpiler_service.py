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
Qiskit IBM Transpiler (:mod:`qiskit_ibm_transpiler.transpiler_service`)
===============================================================================

.. currentmodule:: qiskit_ibm_transpiler.transpiler_service

Classes
=======
.. autosummary::
   :toctree: ../stubs/

   TranspilerService

"""

from typing import Literal

from .types import OptimizationOptions


class TranspilerService:
    """Class for using the cloud transpiler service, which is no longer available.

    .. warning::

        The Qiskit Function ``ibm/transpiler-function`` is deprecated and no longer
        available. Creating this class always raises a ``ValueError``. Transpile
        locally instead: use ``generate_ai_pass_manager`` or the AI transpiler passes
        for AI transpilation, or ``qiskit.transpiler.generate_preset_pass_manager``
        for standard transpilation.

    :param optimization_level: The optimization level to use during the transpilation. There are 4 optimization levels ranging from 0 to 3, where 0 is intended for not performing any optimizations and 3 spends the most effort to optimize the circuit.
    :type optimization_level: int
    :param optimization_preferences: Describe your preferences with a value or a list of values when prioritizing optimization. Allowed options: noise, n_cnots, n_gates, cnot_layers, layers.
    :type optimization_preferences: str | list[str], optional
    :param ai: Specifies if the transpilation should use AI or not, defaults to True.
    :type ai: str, optional
    :param coupling_map: A list of pairs that represents physical links between qubits.
    :type coupling_map: list[list[int]], optional
    :param backend_name: Name of the backend used for doing the transpilation.
    :type backend_name: str, optional
    :param qiskit_transpile_options: Other options to transpile with qiskit.
    :type qiskit_transpile_options: dict, optional
    :param ai_layout_mode: Specifies how to handle the layout selection. There are 3 layout modes: keep (respects the layout set by the previous transpiler passes), improve (uses the layout set by the previous transpiler passes as a starting point) and optimize (ignores previous layout selections).
    :type ai_layout_mode: str, optional
    """

    def __init__(
        self,
        optimization_level: int,
        ai: Literal["true", "false", "auto"] = "true",
        coupling_map: list[list[int]] | None = None,
        backend_name: str | None = None,
        qiskit_transpile_options: dict = None,
        ai_layout_mode: str = None,
        optimization_preferences: (
            OptimizationOptions | list[OptimizationOptions] | None
        ) = None,
        use_fractional_gates: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the instance."""
        raise ValueError(
            "The Qiskit Function 'ibm/transpiler-function' is deprecated and no longer available, "
            "so TranspilerService cannot transpile circuits in the cloud. Transpile locally instead: "
            "use generate_ai_pass_manager or the AI transpiler passes for AI transpilation, "
            "or qiskit.transpiler.generate_preset_pass_manager for standard transpilation. "
            "More info can be found here: https://quantum.cloud.ibm.com/docs/en/guides/ai-transpiler-passes"
        )
