# -*- coding: utf-8 -*-

# (C) Copyright 2022 IBM. All Rights Reserved.


"""Transpile utils"""
from qiskit import QuantumCircuit
from networkx.algorithms import isomorphism


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
    coupling_map: list[list[int]],
):
    """Check whether a synthesized circuit follows a coupling map and respects topology"""
    return check_transpiling(circuit, coupling_map)


def find_symmetries(G):
    return sorted(
        [d[i] for i in range(G.number_of_nodes())]
        for d in isomorphism.GraphMatcher(G, G).subgraph_isomorphisms_iter()
    )
