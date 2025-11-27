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

"""Replace each sequence of Clifford, Linear Function or Permutation gates by a single block of these types of gate."""

from collections import defaultdict
from functools import partial
from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.library import LinearFunction, PermutationGate
from qiskit.converters import circuit_to_dag, dag_to_dagdependency, dagdependency_to_dag
from qiskit.dagcircuit import DAGDepNode, DAGOpNode
from qiskit.dagcircuit.collect_blocks import BlockCollector
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.quantum_info.operators import Clifford
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
    CollectAndCollapse,
    collapse_to_operation,
)
from qiskit.transpiler.passes.utils import control_flow

CLIFFORD_MAX_BLOCK_SIZE = 9
LINEAR_MAX_BLOCK_SIZE = 9
PERMUTATION_MAX_BLOCK_SIZE = 12


clifford_gate_names = [
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "cx",
    "cy",
    "cz",
    "swap",
    "clifford",
    "linear_function",
    "pauli",
]

linear_gate_names = ["cx", "swap", "linear_function"]

permutation_gate_names = ["swap"]

pn_gate_names = [
    "cx",
    "swap",
    "cz",
    "h",
    "s",
    "sdg",
    "sx",
    "sxdg",
    "x",
    "y",
    "z",
    "rz",
    "rx",
    "ry",
    "rzz",
    "rxx",
    "ryy",
    "rzz",
    "rxx",
    "ryy",
    "rzx",
    "rzy",
    "ryx",
]


class Flatten(TransformationPass):
    """Optimized version that pre-filters and batches operations"""

    def __init__(self, node_names):
        super().__init__()
        self.node_names = frozenset(node_names)
        self._circuit_dag_cache = {}

    def run(self, dag: DAGCircuit):
        # Pre-filter nodes by name efficiently
        candidate_nodes = []
        for name in self.node_names:
            candidate_nodes.extend(dag.named_nodes(name))

        if not candidate_nodes:
            return dag  # Early exit if no candidates

        # Batch process: collect all substitutions
        substitutions = []

        for node in candidate_nodes:
            circuit = self._extract_circuit_from_node(node)
            if circuit is not None:
                substitutions.append((node, circuit_to_dag(circuit)))

        # Apply all substitutions
        for node, substitute_dag in substitutions:
            dag.substitute_node_with_dag(node, substitute_dag)

        return dag

    def _extract_circuit_from_node(self, node):
        """Extract QuantumCircuit from node with minimal overhead"""
        try:
            params = node.op.params
            # Direct index access is faster than len() check for expected case
            return (
                params[1]
                if len(params) > 1 and isinstance(params[1], QuantumCircuit)
                else None
            )
        except (AttributeError, IndexError):
            return None


_flatten_cliffords = Flatten(("clifford", "Clifford"))
_flatten_linearfunctions = Flatten(("linear_function", "Linear_function"))
_flatten_permutations = Flatten(("permutation", "Permutation"))
_flatten_paulinetworks = Flatten(("paulinetwork", "PauliNetwork"))


class GreedyBlockCollector(BlockCollector):
    def __init__(self, dag, max_block_size):
        super().__init__(dag)
        # TODO: remove once requirement is set to qiskit>=2
        self.max_block_size = max_block_size

        # Precompute adjacency structures for better performance
        # Renamed to be more clear about what it contains based on direction
        self._next_nodes_cache = defaultdict(list)

        if self.is_dag_dependency:
            self._precompute_from_dag_dependency()
        else:
            self._precompute_from_dag_circuit()

    def _precompute_from_dag_dependency(self):
        """Precompute from DAG dependency using rustworkx on _multi_graph"""
        graph = self.dag._multi_graph
        node_indices = graph.node_indices()

        for node_id in node_indices:
            node_obj = graph[node_id]  # Get the actual node object

            if self._collect_from_back:
                next_node_ids = list(self.dag.direct_predecessors(node_id))
            else:
                next_node_ids = list(self.dag.direct_successors(node_id))

            # Convert node IDs to node objects for consistency
            next_node_objs = [graph[nid] for nid in next_node_ids]

            # Use node object as key (consistent with circuit path)
            self._next_nodes_cache[node_obj] = next_node_objs

    def _precompute_from_dag_circuit(self):
        """Precompute from regular DAGCircuit using standard methods"""
        # Get all DAGOpNodes
        all_nodes = [node for node in self.dag.nodes() if isinstance(node, DAGOpNode)]

        for node in all_nodes:
            if self._collect_from_back:
                # When collecting backwards, get predecessors
                next_nodes = [
                    pred
                    for pred in self.dag.predecessors(node)
                    if isinstance(pred, DAGOpNode)
                ]
            else:
                # When collecting forwards, get successors
                next_nodes = [
                    succ
                    for succ in self.dag.successors(node)
                    if isinstance(succ, DAGOpNode)
                ]

            self._next_nodes_cache[node] = next_nodes

    def collect_matching_block(
        self, filter_fn: Callable, max_block_width: int | None = None
    ) -> list[DAGOpNode | DAGDepNode]:
        """Iteratively collects the largest block of input nodes (that is, nodes with
        ``_in_degree`` equal to 0) that match a given filtering function.
        Examples of this include collecting blocks of swap gates,
        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,
        blocks of two-qubit gates, etc.  Here 'iteratively' means that once a node is collected,
        the ``_in_degree`` of each of its immediate successor is decreased by 1, allowing more nodes
        to become input and to be eligible for collecting into the current block.
        Returns the block of collected nodes.
        """
        # TODO: remove once requirement is set to qiskit>=2
        if max_block_width is None:
            max_block_width = self.max_block_size

        current_block = []
        current_block_qargs = set()

        unprocessed_pending_nodes = list(self._pending_nodes)
        self._pending_nodes = []

        while unprocessed_pending_nodes:
            node = unprocessed_pending_nodes.pop()

            # Early continue for barriers
            if isinstance(node.op, Barrier):
                continue

            if max_block_width is not None:
                # Create a new set to test the width
                test_qargs = current_block_qargs.copy()
                test_qargs.update(node.qargs)
                width_within_budget = len(test_qargs) <= max_block_width
            else:
                width_within_budget = True

            if filter_fn(node) and width_within_budget:
                current_block.append(node)

                if max_block_width is not None:
                    current_block_qargs.update(node.qargs)

                next_nodes = self._next_nodes_cache.get(node, [])

                # If cache miss, fall back to original method for safety
                if not next_nodes and node in self._next_nodes_cache:
                    # Node is in cache but has no successors - this is correct
                    pass
                elif not next_nodes:
                    if hasattr(self, "_direct_succs"):
                        next_nodes = self._direct_succs(node)
                    else:
                        # Fallback to DAG methods
                        if self._collect_from_back:
                            next_nodes = [
                                n
                                for n in self.dag.predecessors(node)
                                if isinstance(n, (DAGOpNode, DAGDepNode))
                            ]
                        else:
                            next_nodes = [
                                n
                                for n in self.dag.successors(node)
                                if isinstance(n, (DAGOpNode, DAGDepNode))
                            ]

                # Process all next nodes
                for next_node in next_nodes:
                    if next_node in self._in_degree:  # Safety check
                        self._in_degree[next_node] -= 1
                        if self._in_degree[next_node] == 0:
                            unprocessed_pending_nodes.append(next_node)
            else:
                self._pending_nodes.append(node)

        return current_block


class BlockChecker:
    def __init__(self, gates, block_class):
        self.gates = gates
        self.block_class = block_class
        self.current_set = set()

    def select(self, node):
        """Decides if the node should be added to the block."""
        return (
            node.op.name in self.gates and getattr(node.op, "condition", None) is None
        )

    def collapse(self, circuit):
        """Construcs an Gate object for the block."""
        self.current_set = set()
        return self.block_class(circuit)


class CliffordInstruction(Instruction):
    def __init__(self, circuit):
        circuit = _flatten_cliffords(circuit)
        super().__init__(
            name="Clifford",
            num_qubits=circuit.num_qubits,
            num_clbits=0,
            params=[Clifford(circuit), circuit],
        )


class PauliNetworkInstruction(Instruction):
    def __init__(self, circuit):
        circuit = _flatten_paulinetworks(circuit)
        super().__init__(
            name="PauliNetwork",
            num_qubits=circuit.num_qubits,
            num_clbits=0,
            params=[None, circuit],
        )


def construct_permutation_gate(circuit):
    circuit = _flatten_permutations(circuit)
    return PermutationGate(LinearFunction(circuit).permutation_pattern())


def construct_linearfunction_gate(circuit):
    circuit = _flatten_linearfunctions(circuit)
    return LinearFunction(circuit)


class RepeatedCollectAndCollapse(CollectAndCollapse):
    def __init__(
        self,
        block_checker: BlockChecker,
        do_commutative_analysis=True,
        split_blocks=True,
        min_block_size=2,
        max_block_size=1000,
        split_layers=False,
        collect_from_back=False,
        num_reps=10,
    ):
        collect_function = lambda dag: GreedyBlockCollector(  # noqa:E731
            dag,
            max_block_size,  # TODO: remove max_block_size once requirement is set to qiskit>=2
        ).collect_all_matching_blocks(
            filter_fn=block_checker.select,
            split_blocks=split_blocks,
            min_block_size=min_block_size,
            split_layers=split_layers,
            collect_from_back=collect_from_back,
            # max_block_width=max_block_size, # TODO: uncomment once requirement is set to qiskit>=2
        )
        collapse_function = partial(
            collapse_to_operation, collapse_function=block_checker.collapse
        )
        super().__init__(
            collect_function=collect_function,
            collapse_function=collapse_function,
            do_commutative_analysis=do_commutative_analysis,
        )
        self.num_reps = num_reps
        self.collect_from_back = collect_from_back

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the CollectLinearFunctions pass on `dag`.
        Args:
            dag (DAGCircuit): the DAG to be optimized.
        Returns:
            DAGCircuit: the optimized DAG.
        """

        # If the option commutative_analysis is set, construct DAGDependency from the given DAGCircuit.
        if self.do_commutative_analysis:
            dag = dag_to_dagdependency(dag)

        for _ in range(self.num_reps):
            # call collect_function to collect blocks from DAG
            blocks = self.collect_function(dag)

            # call collapse_function to collapse each block in the DAG
            self.collapse_function(dag, blocks)

        # If the option commutative_analysis is set, construct back DAGCircuit from DAGDependency.
        if self.do_commutative_analysis:
            dag = dagdependency_to_dag(dag)

        return dag


class CollectCliffords(RepeatedCollectAndCollapse):
    """CollectCliffords(do_commutative_analysis: bool = True, min_block_size: int = 2, max_block_size: int = CLIFFORD_MAX_BLOCK_SIZE, collect_from_back: bool = False, num_reps: int = 10)

    Collects Clifford blocks as `Instruction` objects and stores the original sub-circuit to compare against it after synthesis.

    :param do_commutative_analysis: Enable or disable commutative analysis, defaults to True
    :type do_commutative_analysis: bool, optional
    :param min_block_size: Set the minimum size for blocks generated during the collect Cliffords pass, defaults to 2.
    :type min_block_size: int, optional
    :param max_block_size: Set the maximum size for blocks generated during the collect Cliffords pass, defaults to 9.
    :type max_block_size: int, optional
    :param collect_from_back: Specify if collect blocks in reverse order or not, defaults to False.
    :type collect_from_back: bool, optional
    :param num_reps: Specify how many times to repeat the optimization process, defaults to 10.
    :type num_reps: int, optional
    """

    def __init__(
        self,
        do_commutative_analysis=True,
        min_block_size=2,
        max_block_size=CLIFFORD_MAX_BLOCK_SIZE,
        collect_from_back=False,
        num_reps=10,
    ):
        super().__init__(
            BlockChecker(
                gates=clifford_gate_names,
                block_class=CliffordInstruction,
            ),
            do_commutative_analysis=do_commutative_analysis,
            split_blocks=True,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            split_layers=False,
            collect_from_back=collect_from_back,
            num_reps=num_reps,
        )


class CollectLinearFunctions(RepeatedCollectAndCollapse):
    """CollectLinearFunctions(do_commutative_analysis: bool = True, min_block_size: int = 4, max_block_size: int = LINEAR_MAX_BLOCK_SIZE, collect_from_back: bool = False, num_reps: int = 10)

    Collects blocks of `SWAP` and `CX` as `LinearFunction` objects and stores the original sub-circuit to compare against it after synthesis.

    :param do_commutative_analysis: Enable or disable commutative analysis, defaults to True
    :type do_commutative_analysis: bool, optional
    :param min_block_size: Set the minimum size for blocks generated during the collect linear functions pass, defaults to 4.
    :type min_block_size: int, optional
    :param max_block_size: Set the maximum size for blocks generated during the collect linear functions pass, defaults to 9.
    :type max_block_size: int, optional
    :param collect_from_back: Specify if collect blocks in reverse order or not, defaults to False.
    :type collect_from_back: bool, optional
    :param num_reps: Specify how many times to repeat the optimization process, defaults to 10.
    :type num_reps: int, optional
    """

    def __init__(
        self,
        do_commutative_analysis=True,
        min_block_size=4,
        max_block_size=LINEAR_MAX_BLOCK_SIZE,
        collect_from_back=False,
        num_reps=10,
    ):
        super().__init__(
            BlockChecker(
                gates=linear_gate_names,
                block_class=construct_linearfunction_gate,
            ),
            do_commutative_analysis=do_commutative_analysis,
            split_blocks=True,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            split_layers=False,
            collect_from_back=collect_from_back,
            num_reps=num_reps,
        )


class CollectPermutations(RepeatedCollectAndCollapse):
    """CollectPermutations(do_commutative_analysis: bool = True, min_block_size: int = 4, max_block_size: int = PERMUTATION_MAX_BLOCK_SIZE, collect_from_back: bool = False, num_reps: int = 10)

    Collects blocks of `SWAP` circuits as `Permutations`.

    :param do_commutative_analysis: Enable or disable commutative analysis, defaults to True
    :type do_commutative_analysis: bool, optional
    :param min_block_size: Set the minimum size for blocks generated during the collect permutations pass, defaults to 4.
    :type min_block_size: int, optional
    :param max_block_size: Set the maximum size for blocks generated during the collect permutations pass, defaults to 12.
    :type max_block_size: int, optional
    :param collect_from_back: Specify if collect blocks in reverse order or not, defaults to False.
    :type collect_from_back: bool, optional
    :param num_reps: Specify how many times to repeat the optimization process, defaults to 10.
    :type num_reps: int, optional
    """

    def __init__(
        self,
        do_commutative_analysis=True,
        min_block_size=4,
        max_block_size=PERMUTATION_MAX_BLOCK_SIZE,
        collect_from_back=False,
        num_reps=10,
    ):
        super().__init__(
            BlockChecker(
                gates=permutation_gate_names,
                block_class=construct_permutation_gate,
            ),
            do_commutative_analysis=do_commutative_analysis,
            split_blocks=True,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            split_layers=False,
            collect_from_back=collect_from_back,
            num_reps=num_reps,
        )


class CollectPauliNetworks(RepeatedCollectAndCollapse):
    """CollectPauliNetworks(do_commutative_analysis: bool = True, min_block_size: int = 4, max_block_size: int = 6, collect_from_back: bool = False, num_reps: int = 10)

    Collects PauliNetworks blocks as `Instruction` objects and stores the original sub-circuit to compare against it after synthesis.

    :param do_commutative_analysis: Enable or disable commutative analysis, defaults to True
    :type do_commutative_analysis: bool, optional
    :param min_block_size: Set the minimum size for blocks generated during the collect PauliNetworks pass, defaults to 4.
    :type min_block_size: int, optional
    :param max_block_size: Set the maximum size for blocks generated during the collect PauliNetworks pass, defaults to 6.
    :type max_block_size: int, optional
    :param collect_from_back: Specify if collect blocks in reverse order or not, defaults to False.
    :type collect_from_back: bool, optional
    :param num_reps: Specify how many times to repeat the optimization process, defaults to 10.
    :type num_reps: int, optional
    """

    def __init__(
        self,
        do_commutative_analysis=True,
        min_block_size=4,
        max_block_size=6,
        collect_from_back=False,
        num_reps=10,
    ):
        super().__init__(
            BlockChecker(
                gates=pn_gate_names,
                block_class=PauliNetworkInstruction,
            ),
            do_commutative_analysis=do_commutative_analysis,
            split_blocks=True,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            split_layers=False,
            collect_from_back=collect_from_back,
            num_reps=num_reps,
        )
