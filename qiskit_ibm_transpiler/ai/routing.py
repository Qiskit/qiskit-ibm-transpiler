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

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, TranspilerError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout

from qiskit_ibm_transpiler.wrappers import AIRoutingAPI


class AIRouting(TransformationPass):
    """AIRouting(backend_name: str | None = None, coupling_map: list[list[int]] | None = None, optimization_level: int = 2, layout_mode: str = "OPTIMIZE")

    The `AIRouting` pass acts both as a layout stage and a routing stage.

    :param backend_name: Name of the backend used for doing the transpilation.
    :type backend_name: str, optional
    :param coupling_map: A list of pairs that represents physical links between qubits.
    :type coupling_map: list[list[int]], optional
    :param optimization_level: With a range from 1 to 3, determines the computational effort to spend in the process (higher usually gives better results but takes longer), defaults to 2.
    :type optimization_level: int
    :param layout_mode: Specifies how to handle the layout selection. There are 3 layout modes: keep (respects the layout set by the previous transpiler passes), improve (uses the layout set by the previous transpiler passes as a starting point) and optimize (ignores previous layout selections), defaults to `OPTIMIZE`.
    :type layout_mode: str
    """

    def __init__(
        self,
        backend_name=None,
        coupling_map=None,
        optimization_level: int = 2,
        layout_mode: str = "OPTIMIZE",
        **kwargs,
    ):
        super().__init__()
        if backend_name is not None and coupling_map is not None:
            raise ValueError(
                f"ERROR. Both backend_name and coupling_map were specified as options. Please just use one of them."
            )
        if backend_name is not None:
            self.backend = backend_name
        elif coupling_map is not None:
            if isinstance(coupling_map, CouplingMap):
                self.backend = list(coupling_map.get_edges())
            elif isinstance(coupling_map, list):
                self.backend = coupling_map
            else:
                raise ValueError(
                    f"ERROR. coupling_map should either be a list of int tuples or a Qiskit CouplingMap object."
                )
        else:
            raise ValueError(f"ERROR. Either backend_name OR coupling_map must be set.")

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
        self.service = AIRoutingAPI(**kwargs)

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
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("AIRouting runs on physical circuits only")

        # Pass dag to circuit for sending to AIRouting
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
            qc,
            self.backend,
            self.optimization_level,
            False,
            self.layout_mode,
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
    target_dag._global_phase = dag._global_phase
    target_dag.duration = dag.duration
    target_dag.unit = dag.unit
    target_dag.metadata = dag.metadata
    target_dag._key_cache = dag._key_cache

    return target_dag
