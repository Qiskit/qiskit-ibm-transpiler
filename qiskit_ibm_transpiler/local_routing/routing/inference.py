# -*- coding: utf-8 -*-

# (C) Copyright 2022 IBM. All Rights Reserved.


"""Routing helper module"""
from pathlib import Path

from qiskit_ibm_transpiler import qiskit_ibm_transpiler_rs

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from ..utils.rust_qc_utils import MakeBlocks, qc_to_rust, rust_to_qc

from ..utils.layouts import LayoutIterTypes, get_layout_iter

# TODO: remove this DEFAULT MODEL once we have a way to download the model from HF
DEFAULT_MODEL_PATH = str(Path(__file__).parent / "models" / "routing_model.safetensors")


LAYOUT_ITER_TYPE = {
    "optimize": LayoutIterTypes.Sabre,
    "improve": LayoutIterTypes.Improve,
}


class RoutingInference:
    _routing = None
    _model_path = None

    def __init__(self, model_path=None):
        self.make_blocks = MakeBlocks()
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if RoutingInference._routing is None or RoutingInference._model_path != model_path:
            RoutingInference._routing = qiskit_ibm_transpiler_rs.CircuitRouting(model_path)
            RoutingInference._model_path = model_path
        self.routing = RoutingInference._routing

    def route(
        self,
        circuit,
        coupling_map_edges,
        coupling_map_n_qubits,
        coupling_map_dist_array,
        layout_mode,
        op_params,
        optimization_preferences,
    ):
        # Format circuit for rust
        if circuit.num_qubits < coupling_map_n_qubits:
            circuit = QuantumCircuit(coupling_map_n_qubits).compose(circuit)
        qc_blocks = self.make_blocks(circuit)
        qc_blocks_rust, cargs_dict = qc_to_rust(qc_blocks)

        # Route in rust
        if layout_mode == "keep":
            # Here we dont modify the initial layout
            (rust_qc, (init_layout, _, locations)) = self.routing.route(
                qc_blocks_rust,
                runs=op_params["full_its"],
                coupling_map=coupling_map_edges,
                dists=coupling_map_dist_array,
                err_map=dict(),
                metrics_names=optimization_preferences,
                num_qubits=coupling_map_n_qubits,
            )
        else:
            # Here we improve a provided layout or optimize the layout from scratch
            n_shots = op_params["runs"]
            layout_iter = get_layout_iter(
                LAYOUT_ITER_TYPE[layout_mode], CouplingMap(coupling_map_edges)
            )
            layouts = [ly for _, ly in zip(range(n_shots), layout_iter(circuit))]

            (rust_qc, (init_layout, _, locations)) = self.routing.transpile(
                qc_blocks_rust,
                runs=op_params["full_its"],  # number parallel inner loop runs
                inner_its=op_params["its"],  # number of layout improvement iterations
                its=op_params["reps"],  # number of layout improvement retries
                shots=n_shots,  # number of layout trials
                layout=layouts,
                coupling_map=coupling_map_edges,
                dists=coupling_map_dist_array,
                err_map=dict(),
                metrics_names=optimization_preferences,
                num_qubits=coupling_map_n_qubits,
                max_seconds=op_params["max_time"],
            )

        # Format the result back to Qiskit
        rl_circ = rust_to_qc(
            QuantumCircuit.copy_empty_like(circuit),
            rust_qc,
            list(qc_blocks),
            cargs_dict,
        )

        # Return the result and initial and final layouts
        return rl_circ.decompose("blocks"), init_layout, locations
