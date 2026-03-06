# -*- coding: utf-8 -*-

# (C) Copyright 2022 IBM. All Rights Reserved.


"""Routing helper module"""

import logging
import os
from pathlib import Path

from qiskit_ibm_transpiler import qiskit_ibm_transpiler_rs
from qiskit_ibm_transpiler.hf_models_client import HFInterface
from qiskit_ibm_transpiler.model_bootstrap import _STATIC_SOURCES

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from ..utils.rust_qc_utils import MakeBlocks, qc_to_rust, rust_to_qc

from ..utils.layouts import LayoutIterTypes, get_layout_iter

logger = logging.getLogger(__name__)

LAYOUT_ITER_TYPE = {
    "optimize": LayoutIterTypes.Sabre,
    "improve": LayoutIterTypes.Improve,
}

_REPO_ENV = "QISKIT_TRANSPILER_ROUTING_REPO_ID"
_REVISION_ENV = "QISKIT_TRANSPILER_ROUTING_REVISION"


def _find_safetensors(snapshot_path):
    """Return the first .safetensors file found in *snapshot_path*."""
    match = next(Path(snapshot_path).rglob("*.safetensors"), None)
    if match is None:
        raise FileNotFoundError(
            f"No .safetensors file found in HF snapshot at {snapshot_path}. "
            f"Verify that {_REPO_ENV} points to a valid routing model repository."
        )
    return match


def _download_routing_model():
    """Download the routing model from HuggingFace and return the local file path."""
    defaults = _STATIC_SOURCES.get("routing", {})

    repo_id = os.getenv(_REPO_ENV) or defaults.get("repo_id")
    if not repo_id:
        raise RuntimeError(
            f"Routing model repository not configured. "
            f"Set {_REPO_ENV} or ensure model_sources.yaml contains a 'routing' entry."
        )

    revision = os.getenv(_REVISION_ENV) or defaults.get("revision", "main")

    logger.info(
        "Downloading routing model from %s@%s. "
        "This may take a few minutes on first run...",
        repo_id,
        revision,
    )

    try:
        snapshot_path = HFInterface().download_models(
            repo_id=repo_id, revision=revision
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download routing model from {repo_id}@{revision}. "
            f"Check your network connection and verify the repository exists. "
            f"To override the repository, set {_REPO_ENV}. "
            f"Error: {exc}"
        ) from exc

    return str(_find_safetensors(snapshot_path))


class RoutingInference:
    _routing = None
    _model_path = None

    def __init__(self, model_path=None):
        self.make_blocks = MakeBlocks()
        if model_path is None and RoutingInference._routing is not None:
            self.routing = RoutingInference._routing
            return
        if model_path is None:
            model_path = _download_routing_model()
        if (
            RoutingInference._routing is None
            or RoutingInference._model_path != model_path
        ):
            RoutingInference._routing = qiskit_ibm_transpiler_rs.CircuitRouting(
                model_path
            )
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
            rust_qc, (init_layout, _, locations) = self.routing.route(
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

            rust_qc, (init_layout, _, locations) = self.routing.transpile(
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
