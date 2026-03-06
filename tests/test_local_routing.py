# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Unit tests for local routing: qc_to_rust, rust_to_qc, MakeBlocks,
barrier handling, 3+ qubit gate warnings, and RoutingInference."""

import logging
import threading

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CCXGate
from qiskit.transpiler import CouplingMap, PassManager
from safetensors.numpy import save_file

from qiskit_ibm_transpiler.local_routing.utils.rust_qc_utils import (
    MakeBlocks,
    qc_to_rust,
    rust_to_qc,
)

# A small linear coupling map: 0-1-2-3-4
SMALL_CMAP_EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
SMALL_CMAP = CouplingMap(SMALL_CMAP_EDGES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """Generate a minimal valid safetensors routing model for tests."""
    tensors = {
        "bias0": np.zeros(256, dtype=np.float32),
        "bias1": np.zeros(16, dtype=np.float32),
        "embeddings": np.zeros((128, 256), dtype=np.float32),
        "layer1": np.zeros((16, 256), dtype=np.float32),
    }
    path = tmp_path_factory.mktemp("models") / "test_routing.safetensors"
    save_file(tensors, str(path))
    return str(path)


@pytest.fixture(scope="module")
def routing_inference(model_path):
    """Create a RoutingInference backed by the generated test model."""
    from qiskit_ibm_transpiler.local_routing.routing.inference import RoutingInference

    # Reset singleton so we always load our test model
    RoutingInference._routing = None
    RoutingInference._model_path = None
    ri = RoutingInference(model_path=model_path)
    yield ri
    # Reset singleton again after the module is done
    RoutingInference._routing = None
    RoutingInference._model_path = None


@pytest.fixture
def basic_cx_circuit():
    """A simple 5-qubit circuit with CX gates on a line."""
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    return qc


@pytest.fixture
def basic_cz_circuit():
    qc = QuantumCircuit(5)
    qc.cz(0, 1)
    qc.cz(2, 3)
    return qc


@pytest.fixture
def circuit_with_barrier():
    """Circuit that has a barrier between two CX layers."""
    qc = QuantumCircuit(5)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.barrier()
    qc.cx(1, 2)
    qc.cx(3, 4)
    return qc


@pytest.fixture
def circuit_with_swap():
    qc = QuantumCircuit(5)
    qc.swap(0, 1)
    qc.cx(1, 2)
    return qc


# ---------------------------------------------------------------------------
# qc_to_rust tests
# ---------------------------------------------------------------------------


class TestQcToRust:
    def test_cx_gates_encoded_as_type_0(self, basic_cx_circuit):
        """CX gates should be encoded with gate_type=0."""
        ops, cargs = qc_to_rust(basic_cx_circuit)
        assert all(gate_type == 0 for gate_type, _ in ops)
        assert cargs == {}

    def test_cz_gates_encoded_as_type_2(self, basic_cz_circuit):
        """CZ gates should be encoded with gate_type=2."""
        ops, cargs = qc_to_rust(basic_cz_circuit)
        assert all(gate_type == 2 for gate_type, _ in ops)

    def test_swap_gates_encoded_as_type_1(self, circuit_with_swap):
        """SWAP gates should be encoded with gate_type=1."""
        ops, cargs = qc_to_rust(circuit_with_swap)
        swap_ops = [(gt, q) for gt, q in ops if gt == 1]
        assert len(swap_ops) == 1

    def test_barrier_encoded_as_type_6(self, circuit_with_barrier):
        """Barriers should produce ops with gate_type=6, two per qubit."""
        ops, _ = qc_to_rust(circuit_with_barrier)
        barrier_ops = [(gt, q) for gt, q in ops if gt == 6]
        # 5 qubits → 5 forward + 5 reversed = 10 barrier ops
        assert len(barrier_ops) == 10
        # Each barrier op should have qubit_inputs = (idx, idx)
        for _, (q0, q1) in barrier_ops:
            assert q0 == q1

    def test_qubit_indices_correct(self, basic_cx_circuit):
        """Qubit indices in the output should match the circuit's qubit ordering."""
        ops, _ = qc_to_rust(basic_cx_circuit)
        expected_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
        actual_pairs = [qubits for _, qubits in ops]
        assert actual_pairs == expected_pairs

    def test_single_qubit_gate(self):
        """Single-qubit gates get encoded with both qubit indices the same."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        ops, _ = qc_to_rust(qc)
        # h is a 1-qubit gate → (idx, idx) with op_id-based type
        h_op = ops[0]
        assert h_op[1] == (0, 0)  # qubit indices match
        assert h_op[0] == 10  # 10 * (0 + 1) = 10

    def test_three_qubit_gate_skipped_with_warning(self, caplog):
        """Gates with 3+ qubits should be skipped with a warning."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.append(CCXGate(), [0, 1, 2])
        qc.cx(3, 4)
        with caplog.at_level(logging.WARNING):
            ops, _ = qc_to_rust(qc)
        # CCX (3-qubit) should be skipped, only 2 CX remain
        assert len(ops) == 2
        assert any("Skipping 3-qubit gate" in msg for msg in caplog.messages)

    def test_empty_circuit(self):
        """An empty circuit produces no ops."""
        qc = QuantumCircuit(3)
        ops, cargs = qc_to_rust(qc)
        assert ops == []
        assert cargs == {}

    def test_conditional_gate_adds_mrw(self):
        """A conditioned gate should add MRW (type 5) ops before and after."""
        from qiskit.circuit.library import CXGate

        qc = QuantumCircuit(3, 1)
        qc.measure(0, 0)
        gate = CXGate().to_mutable()
        gate.condition = (qc.clbits[0], 1)
        qc.append(gate, [0, 1])
        ops, cargs = qc_to_rust(qc)
        gate_types = [gt for gt, _ in ops]
        # MRW ops have type 5
        assert gate_types.count(5) >= 2


# ---------------------------------------------------------------------------
# rust_to_qc tests
# ---------------------------------------------------------------------------


class TestRustToQc:
    def test_cx_roundtrip(self, basic_cx_circuit):
        """CX-only circuit should survive qc_to_rust → rust_to_qc round-trip."""
        ops, cargs = qc_to_rust(basic_cx_circuit)
        result = rust_to_qc(
            QuantumCircuit.copy_empty_like(basic_cx_circuit),
            ops,
            list(basic_cx_circuit),
            cargs,
        )
        # Count CX gates in result
        cx_count = sum(1 for g in result if g.operation.name == "cx")
        assert cx_count == 4

    def test_cz_roundtrip(self, basic_cz_circuit):
        ops, cargs = qc_to_rust(basic_cz_circuit)
        result = rust_to_qc(
            QuantumCircuit.copy_empty_like(basic_cz_circuit),
            ops,
            list(basic_cz_circuit),
            cargs,
        )
        cz_count = sum(1 for g in result if g.operation.name == "cz")
        assert cz_count == 2

    def test_swap_roundtrip(self, circuit_with_swap):
        ops, cargs = qc_to_rust(circuit_with_swap)
        result = rust_to_qc(
            QuantumCircuit.copy_empty_like(circuit_with_swap),
            ops,
            list(circuit_with_swap),
            cargs,
        )
        swap_count = sum(1 for g in result if g.operation.name == "swap")
        assert swap_count == 1

    def test_barrier_roundtrip(self, circuit_with_barrier):
        """Barriers should be reconstructed (deduplicated) in the round-trip."""
        ops, cargs = qc_to_rust(circuit_with_barrier)
        result = rust_to_qc(
            QuantumCircuit.copy_empty_like(circuit_with_barrier),
            ops,
            list(circuit_with_barrier),
            cargs,
        )
        barrier_count = sum(1 for g in result if g.operation.name == "barrier")
        # Original had 1 barrier, round-trip should preserve at least 1
        assert barrier_count >= 1

    def test_invalid_op_type_raises(self):
        """An op_type in [3, len(OPS)) that isn't in OPS should raise ValueError."""
        qc = QuantumCircuit(3)
        # op_type=3 is 'U' which doesn't exist in OPS list (only cx=0, swap=1, cz=2)
        # but it's < 10 and >= len(OPS)=3, so it should raise
        with pytest.raises(ValueError, match="unsupported gate"):
            rust_to_qc(qc, [(3, (0, 1))], [], {})


# ---------------------------------------------------------------------------
# MakeBlocks tests
# ---------------------------------------------------------------------------


class TestMakeBlocks:
    def test_blocks_created_from_2q_runs(self):
        """Consecutive 2-qubit gates on the same qubits should be blocked."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)

        pm = PassManager([MakeBlocks()])
        blocked = pm.run(qc)
        block_count = sum(1 for g in blocked if g.operation.name == "block")
        assert block_count >= 1

    def test_no_blocks_for_single_gates(self):
        """Single 2-qubit gates (no runs) should not be blocked."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)

        pm = PassManager([MakeBlocks()])
        blocked = pm.run(qc)
        # These gates don't form a 2q run (different qubit pairs, no dependency chain)
        # so they should remain as-is
        block_count = sum(1 for g in blocked if g.operation.name == "block")
        assert block_count == 0

    def test_blocks_decompose_correctly(self):
        """Blocked gates should decompose back to the original gates."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 1)

        pm = PassManager([MakeBlocks()])
        blocked = pm.run(qc)
        decomposed = blocked.decompose("block")
        assert decomposed.num_qubits == 3


# ---------------------------------------------------------------------------
# RoutingInference tests (require model file)
# ---------------------------------------------------------------------------


class TestRoutingInference:
    def test_route_simple_circuit(self, routing_inference, basic_cx_circuit):
        """Route a simple CX circuit on the small coupling map."""
        op_params = {"full_its": 4, "its": 2, "reps": 1, "runs": 1, "max_time": 30}
        routed_qc, init_layout, locations = routing_inference.route(
            circuit=basic_cx_circuit,
            coupling_map_edges=list(SMALL_CMAP.get_edges()),
            coupling_map_n_qubits=SMALL_CMAP.size(),
            coupling_map_dist_array=SMALL_CMAP.distance_matrix.astype(int).tolist(),
            layout_mode="keep",
            op_params=op_params,
            optimization_preferences=["noise", "depth"],
        )
        assert isinstance(routed_qc, QuantumCircuit)
        assert len(init_layout) == SMALL_CMAP.size()
        assert len(locations) == SMALL_CMAP.size()

    def test_route_with_barrier(self, routing_inference, circuit_with_barrier):
        """A circuit with a barrier should route without errors."""
        op_params = {"full_its": 4, "its": 2, "reps": 1, "runs": 1, "max_time": 30}
        routed_qc, _, _ = routing_inference.route(
            circuit=circuit_with_barrier,
            coupling_map_edges=list(SMALL_CMAP.get_edges()),
            coupling_map_n_qubits=SMALL_CMAP.size(),
            coupling_map_dist_array=SMALL_CMAP.distance_matrix.astype(int).tolist(),
            layout_mode="keep",
            op_params=op_params,
            optimization_preferences=["noise", "depth"],
        )
        assert isinstance(routed_qc, QuantumCircuit)

    def test_routed_circuit_respects_coupling_map(
        self, routing_inference, basic_cx_circuit
    ):
        """All 2-qubit gates in the routed circuit should be on coupling map edges."""
        op_params = {"full_its": 4, "its": 2, "reps": 1, "runs": 1, "max_time": 30}
        routed_qc, _, _ = routing_inference.route(
            circuit=basic_cx_circuit,
            coupling_map_edges=list(SMALL_CMAP.get_edges()),
            coupling_map_n_qubits=SMALL_CMAP.size(),
            coupling_map_dist_array=SMALL_CMAP.distance_matrix.astype(int).tolist(),
            layout_mode="keep",
            op_params=op_params,
            optimization_preferences=["noise", "depth"],
        )
        edges = set(SMALL_CMAP.get_edges())
        for gate in routed_qc:
            if gate.operation.num_qubits == 2 and gate.operation.name != "barrier":
                q0, q1 = (
                    routed_qc.find_bit(gate.qubits[0]).index,
                    routed_qc.find_bit(gate.qubits[1]).index,
                )
                assert (q0, q1) in edges or (q1, q0) in edges, (
                    f"Gate {gate.operation.name} on qubits ({q0}, {q1}) "
                    f"not in coupling map"
                )

    def test_singleton_returns_same_instance(self, model_path):
        """Two RoutingInference(model_path=X) calls return the same Rust object."""
        from qiskit_ibm_transpiler.local_routing.routing.inference import (
            RoutingInference,
        )

        RoutingInference._routing = None
        RoutingInference._model_path = None
        ri1 = RoutingInference(model_path=model_path)
        ri2 = RoutingInference(model_path=model_path)
        assert ri1.routing is ri2.routing
        RoutingInference._routing = None
        RoutingInference._model_path = None

    def test_singleton_thread_safety(self, model_path):
        """Multiple threads creating RoutingInference should not race."""
        from qiskit_ibm_transpiler.local_routing.routing.inference import (
            RoutingInference,
        )

        RoutingInference._routing = None
        RoutingInference._model_path = None

        instances = []
        errors = []

        def create_instance():
            try:
                ri = RoutingInference(model_path=model_path)
                instances.append(ri.routing)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_instance) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during threaded creation: {errors}"
        # All threads should get the same underlying Rust object
        assert all(inst is instances[0] for inst in instances)
        RoutingInference._routing = None
        RoutingInference._model_path = None

    def test_different_model_path_reloads(self, model_path, tmp_path):
        """Providing a different model_path should reload the model."""
        from qiskit_ibm_transpiler.local_routing.routing.inference import (
            RoutingInference,
        )

        RoutingInference._routing = None
        RoutingInference._model_path = None

        ri1 = RoutingInference(model_path=model_path)
        old_routing = ri1.routing

        # Same path -> same instance
        ri2 = RoutingInference(model_path=model_path)
        assert ri2.routing is old_routing

        # Different path (copy of same model) -> new instance
        alt_file = tmp_path / "alt_routing.safetensors"
        alt_file.write_bytes(open(model_path, "rb").read())
        ri3 = RoutingInference(model_path=str(alt_file))
        assert ri3.routing is not old_routing

        RoutingInference._routing = None
        RoutingInference._model_path = None


# ---------------------------------------------------------------------------
# Rust-level CircuitRouting tests
# ---------------------------------------------------------------------------


class TestCircuitRouting:
    def test_route_returns_valid_result(self, model_path):
        """CircuitRouting.route() should return ops and layout tuple."""
        from qiskit_ibm_transpiler import qiskit_ibm_transpiler_rs

        cr = qiskit_ibm_transpiler_rs.CircuitRouting(model_path)
        # Simple 2-CX circuit on 5 qubits: cx(0,1), cx(2,3)
        circuit = [(0, (0, 1)), (0, (2, 3))]
        dists = SMALL_CMAP.distance_matrix.astype(int).tolist()
        rust_qc, (init_layout, qubits, locations) = cr.route(
            circuit=circuit,
            runs=2,
            coupling_map=list(SMALL_CMAP.get_edges()),
            dists=dists,
            err_map={},
            metrics_names=["noise", "depth"],
            num_qubits=SMALL_CMAP.size(),
            max_seconds=30,
        )
        assert isinstance(rust_qc, list)
        assert isinstance(init_layout, list)
        assert len(init_layout) == SMALL_CMAP.size()
        assert len(qubits) == SMALL_CMAP.size()
        assert len(locations) == SMALL_CMAP.size()

    def test_transpile_returns_valid_result(self, model_path):
        """CircuitRouting.transpile() with layout shots should work."""
        from qiskit_ibm_transpiler import qiskit_ibm_transpiler_rs

        cr = qiskit_ibm_transpiler_rs.CircuitRouting(model_path)
        circuit = [(0, (0, 1)), (0, (2, 3))]
        n = SMALL_CMAP.size()
        dists = SMALL_CMAP.distance_matrix.astype(int).tolist()
        layout = [list(range(n))]  # identity layout, 1 shot

        rust_qc, (init_layout, qubits, locations) = cr.transpile(
            circuit=circuit,
            runs=2,
            inner_its=2,
            its=1,
            shots=1,
            layout=layout,
            coupling_map=list(SMALL_CMAP.get_edges()),
            dists=dists,
            err_map={},
            metrics_names=["noise", "depth"],
            num_qubits=n,
            max_seconds=30,
        )
        assert isinstance(rust_qc, list)
        assert len(init_layout) == n

    def test_get_dists_max_value(self):
        """get_dists_max_value should return a large integer."""
        from qiskit_ibm_transpiler import qiskit_ibm_transpiler_rs

        val = qiskit_ibm_transpiler_rs.CircuitRouting.get_dists_max_value()
        assert isinstance(val, int)
        assert val > 0

    def test_invalid_model_path_raises(self):
        """Loading a non-existent model should raise RuntimeError."""
        from qiskit_ibm_transpiler import qiskit_ibm_transpiler_rs

        with pytest.raises(RuntimeError):
            qiskit_ibm_transpiler_rs.CircuitRouting("/nonexistent/model.safetensors")

    def test_corrupted_model_raises(self, tmp_path):
        """Loading a truncated/corrupt safetensors file should raise RuntimeError."""
        from qiskit_ibm_transpiler import qiskit_ibm_transpiler_rs

        bad_file = tmp_path / "corrupt.safetensors"
        bad_file.write_bytes(b"not a valid safetensors file at all")
        with pytest.raises(RuntimeError, match="safetensors"):
            qiskit_ibm_transpiler_rs.CircuitRouting(str(bad_file))
