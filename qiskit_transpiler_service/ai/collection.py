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

from functools import partial

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.library import LinearFunction, PermutationGate
from qiskit.converters import circuit_to_dag, dag_to_dagdependency, dagdependency_to_dag
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


class Flatten(TransformationPass):
    """Replaces all instructions that contain a circuit with their circuit"""

    def __init__(self, node_names):
        super().__init__()
        self.node_names = node_names

    def run(self, dag: DAGCircuit):
        for node in dag.named_nodes(*self.node_names):
            if (
                hasattr(node.op, "params")
                and len(node.op.params) > 1
                and isinstance(node.op.params[1], QuantumCircuit)
            ):
                dag.substitute_node_with_dag(node, circuit_to_dag(node.op.params[1]))

        return dag


_flatten_cliffords = Flatten(("clifford", "Clifford"))
_flatten_linearfunctions = Flatten(("linear_function", "Linear_function"))
_flatten_permutations = Flatten(("permutation", "Permutation"))

from qiskit.dagcircuit.collect_blocks import BlockCollector


class GreedyBlockCollector(BlockCollector):
    def __init__(self, dag, max_block_size):
        super().__init__(dag)
        self.max_block_size = max_block_size

    def collect_matching_block(self, filter_fn):
        """Iteratively collects the largest block of input nodes (that is, nodes with
        ``_in_degree`` equal to 0) that match a given filtering function, limiting the
        maximum size of the block.
        """
        current_block = []
        unprocessed_pending_nodes = self._pending_nodes
        self._pending_nodes = []
        block_qargs = set()

        # Iteratively process unprocessed_pending_nodes:
        # - any node that does not match filter_fn is added to pending_nodes
        # - any node that match filter_fn is added to the current_block,
        #   and some of its successors may be moved to unprocessed_pending_nodes.
        while unprocessed_pending_nodes:
            node = unprocessed_pending_nodes.pop()
            if isinstance(node.op, Barrier):
                continue

            new_qargs = block_qargs.copy()
            for q in node.qargs:
                new_qargs.add(q)

            if filter_fn(node) and len(new_qargs) <= self.max_block_size:
                current_block.append(node)
                block_qargs = new_qargs

                # update the _in_degree of node's successors
                for suc in self._direct_succs(node):
                    self._in_degree[suc] -= 1
                    if self._in_degree[suc] == 0:
                        unprocessed_pending_nodes.append(suc)
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
        collect_function = lambda dag: GreedyBlockCollector(
            dag, max_block_size
        ).collect_all_matching_blocks(
            filter_fn=block_checker.select,
            split_blocks=split_blocks,
            min_block_size=min_block_size,
            split_layers=split_layers,
            collect_from_back=collect_from_back,
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
