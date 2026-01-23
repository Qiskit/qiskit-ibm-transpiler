# -*- coding: utf-8 -*-

"""Tests for topology hash utility helpers."""

import pytest

from qiskit_ibm_transpiler.utils import compute_topology_hash


def test_compute_topology_hash_matches_networkx():
    nx = pytest.importorskip("networkx")

    env_config = {
        "gateset": [
            ("CX", (0, 1)),
            ("CX", (1, 0)),
            ("CX", (1, 2)),
            ("CX", (2, 1)),
            ("H", (0,)),
        ],
    }

    expected_graph = nx.Graph()
    expected_graph.add_nodes_from(range(3))
    expected_graph.add_edge(0, 1)
    expected_graph.add_edge(1, 2)

    assert compute_topology_hash(env_config) == nx.weisfeiler_lehman_graph_hash(
        expected_graph
    )
