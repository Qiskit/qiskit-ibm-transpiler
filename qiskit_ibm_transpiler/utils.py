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
Utilities (:mod:`qiskit_ibm_transpiler.utils`)
===============================================================================

.. currentmodule:: qiskit_ibm_transpiler.utils

Functions
=========

.. autofunction:: create_random_linear_function
.. autofunction:: get_metrics

"""

import base64
import io
import logging
import os
import struct
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3, qpy
from qiskit.circuit import QuantumCircuit, library
from qiskit.circuit.library import LinearFunction
from qiskit.qpy import common
from qiskit.quantum_info import Clifford
from qiskit.synthesis.linear.linear_matrix_utils import random_invertible_binary_matrix
from qiskit.transpiler.basepasses import TransformationPass
from qiskit_ibm_runtime import QiskitRuntimeService

logger = logging.getLogger(__name__)

QPY_QISKIT_VERSION_MAPPING = {
    "1.4.0": 13,
    "1.3.0": 13,
    "1.3.1": 13,
    "1.3.2": 13,
    "1.3.3": 13,
    "1.2.4": 12,
}


def _get_qpy_version(qiskit_version: str):
    service_qpy_version = QPY_QISKIT_VERSION_MAPPING.get(
        qiskit_version, common.QPY_VERSION
    )
    # Return lower version between local qiskit and service to keep compatibility
    return min(common.QPY_VERSION, service_qpy_version)


def get_metrics(qc: QuantumCircuit) -> Dict[str, int]:
    """Returns a dict with metrics from a QuantumCircuit"""
    qcd = qc.decompose(reps=3)
    return {
        "n_gates": qcd.size(),
        "n_layers": qcd.depth(),
        "n_cnots": qcd.num_nonlocal_gates(),
        "n_layers_cnots": qcd.depth(lambda x: x[0].name == "cx"),
    }


def random_permutation(n_qubits):
    """Generate a random permutation of n_qubits qubits."""
    return np.random.permutation(n_qubits)


def create_random_linear_function(n_qubits: int, seed: int = 123) -> LinearFunction:
    return LinearFunction(random_invertible_binary_matrix(n_qubits, seed=seed))


def random_clifford_from_linear_function(n_qubits: int, seed: int = 123):
    """Generate a random clifford from a random linear function of n_qubits qubits."""

    random_linear = LinearFunction(random_invertible_binary_matrix(n_qubits, seed=seed))
    return Clifford(random_linear)


def to_qasm3_iterative_decomposition(circuit: QuantumCircuit, n_iter: int = 10):
    decomposed_circuit = circuit.copy()
    for reps in range(1, n_iter + 1):
        qasm3_str = qasm3.dumps(decomposed_circuit)
        try:
            qasm3.loads(qasm3_str)
            break
        except qasm3.QASM3ImporterError:
            if reps == n_iter:
                raise qasm3.QASM3ExporterError(
                    f"Circuit couldn't be exported to QASM3, try using decompose() on your circuit"
                )
            decomposed_circuit = circuit.decompose(reps=reps)
    return qasm3_str.replace("\n", " ")


def input_to_qasm(input_circ: Union[QuantumCircuit, str]) -> str:
    if isinstance(input_circ, QuantumCircuit):
        try:
            qasm = qasm2.dumps(input_circ).replace("\n", " ")
        except qasm2.QASM2ExportError:
            qasm = to_qasm3_iterative_decomposition(input_circ)
    elif isinstance(input_circ, str):
        qasm = input_circ.replace("\n", " ")
    else:
        raise TypeError("Input circuits must be QuantumCircuit or qasm string.")
    return qasm


class FixECR(TransformationPass):
    def run(self, dag):
        for node in dag.op_nodes():
            if node.name.startswith("ecr"):
                dag.substitute_node(node, library.ECRGate())
        return dag


def get_circuit_from_qasm(qasm_string: str) -> QuantumCircuit:

    try:
        return get_circuit_from_qasm.fix_ecr(
            qasm2.loads(
                qasm_string,
                custom_instructions=get_circuit_from_qasm.QISKIT_INSTRUCTIONS,
            )
        )
    except qasm2.QASM2ParseError:
        return get_circuit_from_qasm.fix_ecr(qasm3.loads(qasm_string))


class QasmType(str, Enum):
    QASM2 = "QASM2"
    QASM3 = "QASM3"


def get_qasm_from_circuit(
    circuit: Union[QuantumCircuit, None], qasm_format: QasmType = QasmType.QASM2
) -> Union[str, None]:
    """Get the QASM from a circuit and remove unneeded line breaks"""
    if circuit is None:
        return None

    try:
        if qasm_format == QasmType.QASM2:
            logger.info("Convert circuit to QASM2")
            qasm = qasm2.dumps(circuit)
        elif qasm_format == QasmType.QASM3:
            logger.info("Convert circuit to QASM3")
            qasm = to_qasm3_iterative_decomposition(circuit)
        else:
            raise ValueError(f"Unsopported format {qasm_format.name}")

    except qasm2.QASM2ExportError:
        logger.info(
            "Incoming circuit had QASM 2 format, but the transpiled circuit couldn't be serialized with QASM 2. Trying QASM 3."
        )
        qasm = to_qasm3_iterative_decomposition(circuit)
    except qasm3.QASM3ExporterError:
        logger.info(
            "Incoming circuit had QASM 3 format, but the transpiled circuit couldn't be serialized with QASM 3. Trying QASM 2."
        )
        qasm = qasm2.dumps(circuit)

    qasm = qasm.replace("\n", " ").strip()
    return qasm


get_circuit_from_qasm.QISKIT_INSTRUCTIONS = list(qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
get_circuit_from_qasm.QISKIT_INSTRUCTIONS.append(
    qasm2.CustomInstruction("ecr", 0, 2, library.ECRGate)
)
get_circuit_from_qasm.fix_ecr = FixECR()


def embed_clifford(cliff, nq):
    new_cliff = Clifford(QuantumCircuit(nq))
    oq = cliff.num_qubits
    new_cliff.stab_x[:oq, :oq] = cliff.stab_x[:, :]
    new_cliff.stab_z[:oq, :oq] = cliff.stab_z[:, :]
    new_cliff.stab_phase[:oq] = cliff.stab_phase[:]

    new_cliff.destab_x[:oq, :oq] = cliff.destab_x[:, :]
    new_cliff.destab_z[:oq, :oq] = cliff.destab_z[:, :]
    new_cliff.destab_phase[:oq] = cliff.destab_phase[:]

    return new_cliff


def check_synthesized_clifford(
    original_clifford: Clifford, synthesized_qc: QuantumCircuit
):
    """Check whether a synthesized circuit does the same as another original clifford"""
    synthesized_clifford = Clifford(synthesized_qc)
    if original_clifford.num_qubits != synthesized_clifford.num_qubits:
        # We can have the situation in transpiling that the synthesized clifford needs to use
        # more qubits than the original Clifford because of the topology and its shape
        # In that case, we should embed the smaller Clifford in the max N of qubits before comparing both
        max_n_qubits = max(
            original_clifford.num_qubits, synthesized_clifford.num_qubits
        )
        if original_clifford.num_qubits > synthesized_clifford.num_qubits:
            synthesized_clifford = embed_clifford(
                cliff=synthesized_clifford, nq=max_n_qubits
            )
        else:
            original_clifford = embed_clifford(cliff=original_clifford, nq=max_n_qubits)
    return synthesized_clifford == original_clifford


def check_transpiling(circ, cmap):
    """Checks if a given circuit follows a specific coupling map"""
    for cc in circ:
        if cc.operation.num_qubits == 2:
            q_pair = tuple(circ.find_bit(qi).index for qi in cc.qubits)
            if (
                q_pair not in cmap
                and q_pair[::-1] not in cmap
                and list(q_pair) not in cmap
                and list(q_pair[::-1]) not in cmap
            ):
                return False
    return True


def check_topology_synthesized_circuit(
    circuit: QuantumCircuit,
    coupling_map: List[List[int]],
):
    """Check whether a synthesized circuit follows a coupling map and respects topology"""
    return check_transpiling(circuit, coupling_map)


def get_qpy_from_circuit(
    input_circ: Union[QuantumCircuit, List[QuantumCircuit]],
    qiskit_version: Union[str, None] = None,
) -> str:
    if isinstance(input_circ, QuantumCircuit) or isinstance(input_circ, list):
        output_b = io.BytesIO()
        qpy.dump(
            input_circ,
            output_b,
            version=_get_qpy_version(qiskit_version),
        )
        qpy_string = base64.b64encode(output_b.getvalue()).decode("utf-8")
    else:
        raise TypeError(
            f"Input circuits must be QuantumCircuit or list of QuantumCircuits. Type provided: {type(input_circ)}"
        )
    return qpy_string


def get_circuit_from_qpy(qpy_string: str) -> QuantumCircuit:
    return qpy.load(io.BytesIO(base64.b64decode(qpy_string.encode("utf-8"))))[0]


def get_circuits_from_qpy(qpy_string: str) -> List[QuantumCircuit]:
    return qpy.load(io.BytesIO(base64.b64decode(qpy_string.encode("utf-8"))))


def serialize_circuit_to_qpy_or_qasm(
    input_circuit: QuantumCircuit,
    qiskit_version: Union[str, None] = None,
) -> Tuple[Union[str, None], Union[str, None]]:
    qpy_result = None
    qasm_result = None
    try:
        qpy_result = get_qpy_from_circuit(input_circuit, qiskit_version)
    except struct.error:
        qasm_result = input_to_qasm(input_circuit).replace("\n", " ")

    return qpy_result, qasm_result


def serialize_circuits_to_qpy_or_qasm(
    input_circuits: List[QuantumCircuit], qiskit_version: Union[str, None] = None
) -> Tuple[Union[str, None], Union[List[str], None]]:
    qpy_result = None
    qasm_result = None
    try:
        qpy_result = get_qpy_from_circuit(input_circuits, qiskit_version)
    except struct.error:
        qasm_result = [
            input_to_qasm(the_circ).replace("\n", " ") for the_circ in input_circuits
        ]

    return qpy_result, qasm_result


def deserialize_circuit_from_qpy_or_qasm(
    qpy_input: Union[str, None], qasm_input: Union[str, None]
) -> Union[QuantumCircuit, None]:
    if qpy_input:
        return get_circuit_from_qpy(qpy_input)
    elif qasm_input:
        return get_circuit_from_qasm(qasm_input)
    else:
        return None


def get_qiskit_runtime_service():
    args = {}
    if os.getenv("QISKIT_IBM_RUNTIME_API_URL") is not None:
        args = {
            "channel": os.getenv("QISKIT_IBM_CHANNEL"),
            "token": os.getenv("QISKIT_IBM_TOKEN"),
            "url": os.getenv("QISKIT_IBM_RUNTIME_API_URL"),
        }
    return QiskitRuntimeService(**args)
