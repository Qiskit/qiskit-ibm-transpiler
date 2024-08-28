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

import json
import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Dict
from urllib.parse import urljoin

import backoff
import requests
from qiskit.transpiler.exceptions import TranspilerError

logger = logging.getLogger(__name__)


class BackendTaskError(Exception):
    def __init__(self, status: str, msg: str):
        self.status = status
        self.msg = msg


def _get_token_from_system():
    token = os.environ.get("QISKIT_IBM_TOKEN")

    if not token:
        qiskit_file = Path.home() / ".qiskit" / "qiskit-ibm.json"
        if not qiskit_file.exists():
            raise Exception(
                f"Credentials file {qiskit_file} does not exist. Please set env var QISKIT_IBM_TOKEN to access the service, or save your IBM Quantum API token using QiskitRuntimeService. "
                "More info about saving your token using QiskitRuntimeService https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.QiskitRuntimeService#save_account"
            )
        with open(qiskit_file, "r") as _sc:
            creds = json.loads(_sc.read())
        token = creds.get("default-ibm-quantum", {}).get("token")
        if token is None:
            raise Exception(
                f"default-ibm-quantum not found in {qiskit_file}. Please set env var QISKIT_IBM_TOKEN to access the service, or save your IBM Quantum API token using QiskitRuntimeService. "
                "More info about saving your token using QiskitRuntimeService https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.QiskitRuntimeService#save_account"
            )
    return token


class QiskitTranspilerService:
    """A helper class that covers some common basic funcionality for the Qiskit transpiler service"""

    def __init__(
        self,
        path_param=None,
        base_url: str = "https://cloud-transpiler.quantum.ibm.com/",
        token: str = None,
        timeout: int = 300,
    ):
        # If it does not recive URL or token, the function tries to find your Qiskit
        # token from the QISKIT_IBM_TOKEN env var
        # If it couldn't find it, it will try to get it from your ~/.qiskit/qiskit-ibm.json file
        # If it couldn't find it, it fails

        url_with_path = urljoin(base_url, path_param)

        url_env_param = f"{path_param.upper()}_" if path_param else ""
        url_env_var = f"QISKIT_TRANSPILER_SERVICE_{url_env_param}URL"

        self.url = os.environ.get(url_env_var, url_with_path).rstrip("/")

        token = token if token else _get_token_from_system()

        self.timeout = timeout

        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Caller": "qiskit-transpiler-service",
        }

    def get_versions(self):
        url = f"{self.url}/version"
        resp = requests.get(
            url,
            headers=self.headers,
        ).json()

        return resp

    def get_supported_backends(self):
        url = f"{self.url}/backends"
        resp = requests.get(
            url,
            headers=self.headers,
        ).json()

        return resp

    def request_status(self, endpoint, task_id):
        def _giveup(e):
            # Only retry 520 errors
            return e.response.status_code != 520

        @backoff.on_predicate(
            backoff.constant,
            lambda res: res.get("state") not in ["SUCCESS", "FAILURE"],
            jitter=None,
            interval=1,  # TODO: Define by config or circuit?
            max_time=self.timeout,
        )
        @backoff.on_exception(
            backoff.expo,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.JSONDecodeError,
                requests.exceptions.HTTPError,
            ),
            giveup=_giveup,
            max_time=self.timeout,
        )
        def _request_status(self, endpoint, task_id):
            logger.debug(f"Getting status of task {task_id} ...")
            res = requests.get(
                url=f"{self.url}/{endpoint}/{task_id}", headers=self.headers
            )
            res.raise_for_status()
            return res.json()

        return _request_status(self, endpoint, task_id)

    def request_and_wait(self, endpoint: str, body: Dict, params: Dict):
        try:
            return self._request_and_wait(endpoint, body, params)
        except requests.exceptions.HTTPError as exc:
            _raise_transpiler_error_and_log(_get_error_msg_from_response(exc))
        except BackendTaskError as exc:
            error_msg = exc.msg or "Service error."
            _raise_transpiler_error_and_log(error_msg)
        except Exception as exc:
            _raise_transpiler_error_and_log(f"Error: {exc}")

    def _request_and_wait(self, endpoint: str, body: Dict, params: Dict):
        resp = requests.post(
            f"{self.url}/{endpoint}",
            headers=self.headers,
            json=body,
            params=params,
        )
        resp.raise_for_status()
        resp = resp.json()
        task_id = resp.get("task_id")

        result = BackendTaskError(
            status="PENDING",
            msg=f"The background task {task_id} timed out. Try to update the client's timeout config or review your task",
        )

        resp = self.request_status(endpoint, task_id)
        if resp.get("state") == "SUCCESS":
            result = resp.get("result")
        elif resp.get("state") == "FAILURE":
            logger.error("The request FAILED")
            result = BackendTaskError(
                status="FAILURE", msg=f"The background task {task_id} FAILED"
            )

        if isinstance(result, BackendTaskError):
            # TODO: Shall we show this  "The background task 99cf52d2-3942-4ae5-b2a7-d672af7f1216 FAILED" to the user?
            logger.error(f"Failed to get a result for {endpoint}: {result.msg}")
            raise result
        else:
            return result


def _raise_transpiler_error_and_log(msg: str):
    logger.error(msg)
    raise TranspilerError(msg)


def _get_error_msg_from_response(exc: requests.exceptions.HTTPError):
    try:
        resp = exc.response.json()
        detail = resp.get("detail")
        # Default message
        msg = "Internal error."

        if isinstance(detail, str):
            msg = detail
        elif isinstance(detail, list):
            detail_input = detail[0]["input"]
            detail_msg = detail[0]["msg"]

            if detail_input and detail_msg:
                msg = f"Wrong input '{detail_input}'. {detail_msg}"
    except Exception:
        # If something fails decoding the error
        # just show the incoming error
        msg = f"Internal error: {str(exc)}"
    return msg
