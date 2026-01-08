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
"""Transpilation via the transpiler qiskit function"""

import logging
import re
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit_serverless import ServerlessClient

from qiskit_ibm_transpiler.types import OptimizationOptions

from .base import _get_credentials_from_system

# TODO: add actual logging if required, else remove this
logger = logging.getLogger(__name__)


class QiskitTranspilerFunction:
    """A helper class that used the qiskit runtime function interface for transpilation"""

    SERVERLESS_URL = "https://qiskit-serverless.quantum.ibm.com"
    DEFAULT_CHANNEL = "ibm_quantum_platform"
    TRANSPILER_FUNCTION_NAME = "ibm/transpiler-function"

    def __init__(
        self,
        url: str = None,
        token: str = None,
        channel: str = None,
        timeout: int = 300,
        instance: str = None,
        account_name: str = None,
    ):
        """Connects the serverless client and initialized the function"""
        if url is None:
            url = self.SERVERLESS_URL
        # If token or instance is not provided, try to get both from system credentials
        if token is None or instance is None:
            credentials = _get_credentials_from_system(account_name=account_name)
            if token is None:
                token = credentials["token"]
            if instance is None:
                instance = credentials["instance"]
        if channel is None:
            channel = self.DEFAULT_CHANNEL
        self.instance = instance
        logger.debug(
            f"Attempting to get serverless with parameters:\nhost={url}\nchannel={channel}"
        )
        self.serverless = ServerlessClient(
            host=url, channel=channel, token=token, instance=self.instance
        )
        logger.debug(
            f"Attempting to get the transpiler function {self.TRANSPILER_FUNCTION_NAME}"
        )
        self.function = self.serverless.get(self.TRANSPILER_FUNCTION_NAME)
        self.timeout = timeout
        logger.info(
            "QiskitTranspilerFunction initialized successfully, got connection to serverless and the transpilation function"
        )

    def transpile(
        self,
        circuits: list[QuantumCircuit] | QuantumCircuit,
        optimization_level: int,
        backend: str | None = None,
        coupling_map: list[list[int]] | None = None,
        optimization_preferences: (
            OptimizationOptions | list[OptimizationOptions] | None
        ) = None,
        ai: Literal["true", "false", "auto"] = "true",
        qiskit_transpile_options: dict = None,
        ai_layout_mode: str = None,
        use_fractional_gates: bool = False,
        **kwargs,
    ):
        """Calls the transpiler function with the given inputs"""
        # TODO: are optimization_preferences supported in the transpiler function?
        # should they be merged with qiskit_transpile_options?
        # if backend is None, get a default one using the serverless client
        if ai is "false" and optimization_preferences is not None:
            logger.warning(
                "The `optimization_preferences` parameter is only available for `ai` transpilation"
            )
        if len(kwargs) > 0:
            logger.warning(
                f"{len(kwargs)} additional parameters are ignored: {kwargs.keys()}"
            )
        if ai_layout_mode:
            logger.warning(
                f"ai_layout_mode ignored in current version. It will be used in future versions"
            )
        logger.info("About to call the transpilation function")
        logger.debug(f"optimization_level={optimization_level}")
        logger.debug(f"ai={ai}")
        logger.debug(f"ai_layout_mode={ai_layout_mode}")
        logger.debug(f"backend_name={backend}")
        logger.debug(f"transpile_options={qiskit_transpile_options}")
        logger.debug(f"use_fractional_gates={use_fractional_gates}")
        job = self.function.run(
            circuits=circuits,
            optimization_level=optimization_level,
            backend_name=backend,
            coupling_map=coupling_map,
            transpile_options=qiskit_transpile_options,
            use_fractional_gates=use_fractional_gates,
            instance=self.instance,
        )
        logger.debug("Job submitted successfully")
        # TODO: catch exceptions, fail gracefully
        job_result = job.result(maxwait=self.timeout)
        logger.debug(f"Job result obtained: {job_result}")
        if isinstance(job_result, dict) and "transpiled_circuits" in job_result:
            transpiled_circuits = job_result["transpiled_circuits"]
            # TODO: we follow suit with the old service which returns the circuit when there's only one
            # but should it be the case if `circuits` was a list of length 1 and not a `QuantumCircuit`?
            return (
                transpiled_circuits
                if len(transpiled_circuits) > 1
                else transpiled_circuits[0]
            )
        msg = "Unknown error"
        if isinstance(job_result, dict) and "exception" in job_result:
            msg = self.analyze_error_msg(job_result["exception"])
        raise TranspilerError(f"Transpilation failed. Error: {msg}")

    def analyze_error_msg(self, msg):
        try:
            MISSING_BACKEND_ERROR = "The requested backend was not found"
            MISSING_BACKEND_REGEX = "'(\w+)'"
            if re.search(MISSING_BACKEND_ERROR, msg):
                backends = re.findall(MISSING_BACKEND_REGEX, msg)
                return f"The requested backend was not found. Available backends: {backends}"

            return msg

        except Exception:
            return msg
