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

import logging
from typing import Dict, List, Union, Literal

from qiskit import QuantumCircuit

from .wrappers.transpile import TranspileAPI

logger = logging.getLogger(__name__)


class TranspilerService:
    """Class for using the transpiler service.

    :param optimization_level: The optimization level to use during the transpilation. There are 4 optimization levels ranging from 0 to 3, where 0 is intended for not performing any optimizations and 3 spends the most effort to optimize the circuit.
    :type optimization_level: int
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
        coupling_map: Union[List[List[int]], None] = None,
        backend_name: Union[str, None] = None,
        qiskit_transpile_options: Dict = None,
        ai_layout_mode: str = None,
        **kwargs,
    ) -> None:
        """Initializes the instance."""

        self.transpiler_service = TranspileAPI(**kwargs)

        self.backend_name = backend_name
        self.coupling_map = coupling_map
        self.optimization_level = optimization_level
        self.ai = ai
        self.qiskit_transpile_options = qiskit_transpile_options

        if ai_layout_mode is not None:
            if ai_layout_mode.upper() not in ["KEEP", "OPTIMIZE", "IMPROVE"]:
                raise (
                    f"ERROR. Unknown ai_layout_mode: {ai_layout_mode.upper()}. Valid modes: 'KEEP', 'OPTIMIZE', 'IMPROVE'"
                )
            self.ai_layout_mode = ai_layout_mode.upper()
        else:
            self.ai_layout_mode = ai_layout_mode
        super().__init__()

    def run(
        self,
        circuits: Union[List[Union[str, QuantumCircuit]], Union[str, QuantumCircuit]],
    ):
        """Transpile the circuit(s) by calling the service /transpile endpoint.

        Args:
            circuits: circuit(s) to transpile.

        Returns:
            The transpiled circuit(s)

        """
        logger.info(f"Requesting transpile to the service")
        transpile_result = self.transpiler_service.transpile(
            circuits=circuits,
            backend=self.backend_name,
            coupling_map=self.coupling_map,
            optimization_level=self.optimization_level,
            ai=self.ai,
            qiskit_transpile_options=self.qiskit_transpile_options,
            ai_layout_mode=self.ai_layout_mode,
        )
        if transpile_result is None:
            logger.warning("Qiskit IBM Transpiler couldn't transpile the circuit(s)")
            return None

        logger.info("Qiskit IBM Transpiler returned a result")
        return transpile_result
