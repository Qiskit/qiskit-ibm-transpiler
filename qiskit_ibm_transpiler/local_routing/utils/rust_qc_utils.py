# Passes to collect gates into 2q blocks for routing
from qiskit.circuit.instruction import Instruction
from qiskit.dagcircuit.collect_blocks import BlockCollapser
from qiskit.transpiler.basepasses import TransformationPass


class BlockGate(Instruction):
    def __init__(self, block_circuit):
        super().__init__(
            "block", block_circuit.num_qubits, block_circuit.num_clbits, []
        )
        self._block_circuit = block_circuit

    def _define(self):
        self.definition = self._block_circuit


class MakeBlocks(TransformationPass):
    def run(self, dag):
        blocks = [b for b in dag.collect_2q_runs() if len(b) > 1]

        BlockCollapser(dag).collapse_to_operation(blocks, BlockGate)

        return dag


to_blocks = MakeBlocks()

OPS = ["cx", "swap", "cz"]
OPS_DICT = {op: i for i, op in enumerate(OPS)}


def qc_to_rust(qc_blocks):
    ops = []
    cargs_dict = dict()
    has_conditional_gates = any(getattr(gi.operation, "condition", None) is not None for gi in qc_blocks)

    for op_id, gi in enumerate(qc_blocks):
        if gi.operation.name == "barrier":
            # Apply two layers of barrier in both directions
            for q in gi.qubits:
                ops.append(
                    (6, (q._index, q._index))
                )  # Gate type '6' is Barrier for the Rust routing
            for q in reversed(gi.qubits):
                ops.append((6, (q._index, q._index)))
            continue
        if len(gi.qubits) == 2:
            qubit_inputs = tuple(qi._index for qi in gi.qubits)
            if getattr(gi.operation, "condition", None) is None and gi.operation.name in OPS_DICT:
                gate_type = OPS_DICT[gi.operation.name]
            else:
                # if gate is not in dict or is conditioned we choose a generic gate with id
                gate_type = 10 * (op_id + 1)
        elif len(gi.qubits) == 1:
            # This should only happen in some edge cases and in dynamic operations
            qubit_inputs = (gi.qubits[0]._index, gi.qubits[0]._index)
            gate_type = 10 * (op_id + 1)
        else:
            continue

        if len(gi.clbits) > 0:
            cargs_dict[op_id] = gi.clbits

        add_mrw = has_conditional_gates and (
            (getattr(gi.operation, "condition", None) is not None) or (len(gi.clbits) > 0)
        )

        if add_mrw:
            # Here we have to add a MRW gate before the op for the dependencies on the dag
            ops.append(
                (5, (qubit_inputs[0], qubit_inputs[0]))
            )  # Gate type '5' is MRW for the Rust routing

        # here we append the op itself
        ops.append((gate_type, qubit_inputs))

        if add_mrw:
            # Here we have to add a MRW gate after the op for the dependencies on the dag
            ops.append(
                (5, (qubit_inputs[0], qubit_inputs[0]))
            )  # Gate type '5' is MRW for the Rust routing

    return ops, cargs_dict


def rust_to_qc(qc, ops, op_list, cargs_dict):
    # Then add the routed ops
    barrier_qubits = []
    num_barriers = 0
    for op_type, qargs in ops:
        if op_type == 6:  # Collect barrier qubits
            barrier_qubits.append(qargs[0])
            if (
                len(barrier_qubits) == qc.num_qubits
            ):  # Apply barrier when we have all pieces
                if (
                    num_barriers % 2 == 0
                ):  # Only apply the even barriers to remove duplicates
                    qc.barrier(*barrier_qubits)
                barrier_qubits = []
                num_barriers += 1
            elif len(barrier_qubits) > qc.num_qubits:
                pass
            continue

        elif op_type == 5:
            # These are "fake" ops just to introduce dependencies in dag for dynamic ops
            continue

        elif op_type < 10:
            assert op_type < len(
                OPS
            ), f"Found unsuported gate that is not on a block, with id={op_type}"
            if OPS[op_type] == "cx":
                qc.cx(*qargs)
            elif OPS[op_type] == "swap":
                qc.swap(*qargs)
            elif OPS[op_type] == "cz":
                qc.cz(*qargs)

        else:
            op_id = op_type // 10 - 1
            if (len(qargs) == 2) and (qargs[0] == qargs[1]):
                qargs = (qargs[0],)
            qc.append(op_list[op_id].operation, qargs, cargs_dict.get(op_id, tuple()))
    return qc.decompose("block")
