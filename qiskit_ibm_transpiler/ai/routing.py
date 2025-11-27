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

"""Routing and Layout selection with AI"""
# import torch

# torch.set_num_threads(1)

import importlib
import logging
import os

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers.backend import BackendV2 as Backend
from qiskit.transpiler import CouplingMap, TranspilerError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout

from qiskit_ibm_transpiler.types import OptimizationOptions
from qiskit_ibm_transpiler.wrappers import AILocalRouting, AIRoutingAPI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_final_optimization_preferences(
    selected_optimization_preferences: (
        OptimizationOptions | list[OptimizationOptions]
    ) = None,
    backend_name: str = None,
):
    all_optimization_preferences = [
        "noise",
        "cnot_layers",
        "n_cnots",
        "layers",
        "n_gates",
    ]
    optimization_preferences_without_noise = [
        "cnot_layers",
        "n_cnots",
        "layers",
        "n_gates",
    ]

    if not selected_optimization_preferences and not backend_name:
        return optimization_preferences_without_noise
    if not selected_optimization_preferences and backend_name:
        return all_optimization_preferences

    if selected_optimization_preferences and not backend_name:
        if "noise" in selected_optimization_preferences:
            raise ValueError("Error. Cannot optimize by noise without a backend")

    if isinstance(selected_optimization_preferences, list):
        for selected_optimization_preference in selected_optimization_preferences:
            if selected_optimization_preference not in all_optimization_preferences:
                raise ValueError(
                    (
                        f"'{selected_optimization_preference}' is not a valid optimization preference"
                    )
                )

        return selected_optimization_preferences
    else:
        if selected_optimization_preferences not in all_optimization_preferences:
            raise ValueError(
                (
                    f"'{selected_optimization_preferences}' is not a valid optimization preference"
                )
            )

        try:
            optimization_preferences_without_noise.remove(
                selected_optimization_preferences
            )
        except ValueError:
            # If selected_optimization_preferences is not on the list, do nothing (this will be the "noise" usecase)
            pass

        return [
            selected_optimization_preferences
        ] + optimization_preferences_without_noise


class AIRouting(TransformationPass):
    """AIRouting(coupling_map: list[list[int]] | None = None, backend: Backend | None = None, optimization_level: int = 2, layout_mode: str = "OPTIMIZE")

    The `AIRouting` pass acts both as a layout stage and a routing stage.

    :param coupling_map: A list of pairs that represents physical links between qubits.
    :type coupling_map: list[list[int]], optional
    :param backend: Backend used for doing the transpilation.
    :type backend: Backend, optional
    :param optimization_level: With a range from 1 to 3, determines the computational effort to spend in the process (higher usually gives better results but takes longer), defaults to 2.
    :type optimization_level: int
    :param layout_mode: Specifies how to handle the layout selection. There are 3 layout modes: keep (respects the layout set by the previous transpiler passes), improve (uses the layout set by the previous transpiler passes as a starting point) and optimize (ignores previous layout selections), defaults to `OPTIMIZE`.
    :type layout_mode: str
    """

    def __init__(
        self,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        backend: Backend | None = None,
        optimization_level: int = 2,
        layout_mode: str = "OPTIMIZE",
        optimization_preferences: (
            OptimizationOptions | list[OptimizationOptions] | None
        ) = None,
        local_mode: bool = True,
        **kwargs,
    ):
        ai_local_package = "qiskit_ibm_ai_local_transpiler"
        if local_mode:
            if importlib.util.find_spec(ai_local_package) is None:
                raise ImportError(
                    f"For using the local mode you need to install the package '{ai_local_package}'. Read the installation guide for more information"
                )

        if backend and coupling_map:
            raise ValueError(
                f"ERROR. Both backend and coupling_map were specified as options. Please just use one of them."
            )

        if not backend and not coupling_map:
            raise ValueError(
                f"ERROR. One of these options must be set: backend or coupling_map."
            )

        if optimization_level <= 0 or optimization_level > 3:
            raise ValueError(
                f"ERROR. The optimization_level should be a value between 1 and 3."
            )

        super().__init__()

        backend_name = getattr(backend, "name", None)

        self.optimization_preferences = build_final_optimization_preferences(
            optimization_preferences, backend_name
        )

        if coupling_map:
            if isinstance(coupling_map, CouplingMap):
                self.coupling_map = coupling_map
            elif isinstance(coupling_map, list):
                self.coupling_map = CouplingMap(coupling_map)
            else:
                raise ValueError(
                    f"ERROR. coupling_map should either be a list of int tuples or a Qiskit CouplingMap object."
                )
        elif backend:
            is_noise_a_optimization_preference = (
                self.optimization_preferences is not None
                and (
                    self.optimization_preferences == "noise"
                    or "noise" in self.optimization_preferences
                )
            )
            if is_noise_a_optimization_preference:
                # AIRoutingAPI expects a backend_name when trying to optimize by noise
                # TODO: Check if we need the backend for the local mode too
                self.coupling_map = (
                    backend_name if not local_mode else backend.coupling_map
                )
            else:
                self.coupling_map = backend.coupling_map

        self.optimization_level = optimization_level

        if layout_mode is not None and layout_mode.upper() not in [
            "KEEP",
            "OPTIMIZE",
            "IMPROVE",
        ]:
            raise ValueError(
                f"ERROR. Unknown ai_layout_mode: {layout_mode}. Valid modes: 'KEEP', 'OPTIMIZE', 'IMPROVE'"
            )

        self.layout_mode = layout_mode.upper()

        routing_provider = AILocalRouting() if local_mode else AIRoutingAPI(**kwargs)
        self.service = routing_provider

    def run(self, dag):
        """Run the AIRouting pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None
        """
        qc = None
        if _non_physical_circuit_check(dag):
            # Some valid circuits fail this tests, i.e quantum_volume created circuits
            # embedding them in an empty circuit is a workaround
            qc = dag_to_circuit(dag)
            qc = QuantumCircuit(qc.num_qubits).compose(qc)
            dag = circuit_to_dag(qc)
            if _non_physical_circuit_check(dag):
                raise TranspilerError("AIRouting runs on physical circuits only")

        # Pass dag to circuit for sending to AIRouting
        if not qc:
            qc = dag_to_circuit(dag)

        # Remove measurements before sending to AIRouting
        # TODO: Fix this for mid-circuit measurements
        cregs = []
        measurements = []
        if len(qc.cregs) > 0:
            cregs = qc.cregs.copy()
            measurements = [g for g in qc if g.operation.name == "measure"]
            qc.remove_final_measurements(inplace=True)

        routed_qc, init_layout, final_layout = self.service.routing(
            circuit=qc,
            coupling_map=getattr(self, "coupling_map", None),
            optimization_level=self.optimization_level,
            check_result=False,
            layout_mode=self.layout_mode,
            optimization_preferences=self.optimization_preferences,
        )

        # TODO: Improve this
        nq = max(init_layout) + 1
        routed_qc = QuantumCircuit(nq).compose(
            routed_qc, qubits=range(routed_qc.num_qubits)
        )

        # Restore final measurements if they were removed
        if len(measurements) > 0:
            meas_qubits = [final_layout[g.qubits[0]._index] for g in measurements]
            routed_qc.barrier(meas_qubits)
            for creg in cregs:
                routed_qc.add_register(creg)
            for g, q in zip(measurements, meas_qubits):
                routed_qc.append(g.operation, qargs=(q,), cargs=g.clbits)

        # Pass routed circuit to dag
        # TODO: Improve this
        routed_dag = circuit_to_dag(routed_qc)
        if routed_qc.num_qubits > dag.num_qubits():
            new_dag = copy_dag_metadata(dag, routed_dag)
        else:
            new_dag = dag.copy_empty_like()
            new_dag.compose(routed_dag)

        qubits = new_dag.qubits
        input_qubit_mapping = {q: i for i, q in enumerate(qubits)}

        full_initial_layout = init_layout + sorted(
            set(range(len(qubits))) - set(init_layout)
        )
        full_final_layout = final_layout + list(range(len(final_layout), len(qubits)))
        full_final_layout_p = [
            full_final_layout[i] for i in np.argsort(full_initial_layout)
        ]

        initial_layout_qiskit = Layout(dict(zip(full_initial_layout, qubits)))
        final_layout_qiskit = Layout(dict(zip(full_final_layout_p, qubits)))

        self.property_set["layout"] = initial_layout_qiskit
        self.property_set["original_qubit_indices"] = input_qubit_mapping
        self.property_set["final_layout"] = final_layout_qiskit

        return new_dag


def _non_physical_circuit_check(dag):
    """Check if the circuit is non-physical (i.e contains virtual qubits)"""
    if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
        return True
    return False


def add_measurements(circ, qubits):
    circ.add_register(ClassicalRegister(len(qubits)))
    circ.barrier()
    for i, q in enumerate(qubits):
        circ.measure(q, i)
    return circ


def copy_dag_metadata(dag, target_dag):
    """Return a copy of self with the same structure but empty.

    That structure includes:
        * name and other metadata
        * global phase
        * duration
        * all the qubits and clbits, including the registers.

    Returns:
        DAGCircuit: An empty copy of self.
    """
    target_dag.name = dag.name
    target_dag.global_phase = dag.global_phase
    target_dag.duration = dag.duration
    target_dag.unit = dag.unit
    target_dag.metadata = dag.metadata
    # target_dag._key_cache = dag._key_cache

    return target_dag
