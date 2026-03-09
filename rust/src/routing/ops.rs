#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum OpType {
    CX,
    SWAP,
    CZ,
    U,
    CP,
    B(usize, usize),
    MRW,
    BARRIER,
}

pub const OP_TYPES: [OpType; 7] = [
    OpType::CX,
    OpType::SWAP,
    OpType::CZ,
    OpType::U,
    OpType::CP,
    OpType::MRW,
    OpType::BARRIER,
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operation {
    pub id: usize,
    pub op_type: OpType,
    pub qubits: (usize, usize),
}

pub enum MatchingOps {
    RemoveFOp,       // Op cancels FOp, so just remove FOp
    SwapPassthrough, // Op is a Swap and passes through FOp swapping FOp's qargs
    RemoveFOpAndApply(Vec<Operation>),
    FOpAbsorbs, // FOp absorbs Op
    JustAddOp,  // Nothing to simplify here
}

pub const fn op_cost(op_type: OpType) -> usize {
    match op_type {
        OpType::CX => 1,
        OpType::CZ => 1,
        OpType::SWAP => 3,
        OpType::U => 3,
        OpType::CP => 2,
        OpType::B(cost, _op_id) => cost,
        OpType::MRW => 0,
        OpType::BARRIER => 0,
    }
}

pub const fn optype_to_usize(op_type: OpType) -> usize {
    match op_type {
        OpType::CX => 0,
        OpType::SWAP => 1,
        OpType::CZ => 2,
        OpType::U => 3,
        OpType::CP => 4,
        OpType::B(_cost, op_id) => op_id,
        OpType::MRW => 5,
        OpType::BARRIER => 6,
    }
}

fn merge_ops(op: &Operation, fop: &Operation) -> Vec<Operation> {
    match (op.op_type, fop.op_type) {
        // TODO: track what the Us absorb
        (OpType::U, _) => vec![op.clone()],
        (_, OpType::U) => vec![op.clone()],

        // CX & SWAPs
        (OpType::CX, OpType::SWAP) => vec![
            Operation {
                id: fop.id,
                op_type: OpType::CX,
                qubits: (op.qubits.0, op.qubits.1),
            },
            Operation {
                id: op.id,
                op_type: OpType::CX,
                qubits: (op.qubits.1, op.qubits.0),
            },
        ],
        (OpType::SWAP, OpType::CX) => vec![
            Operation {
                id: fop.id,
                op_type: OpType::CX,
                qubits: (op.qubits.0, op.qubits.1),
            },
            Operation {
                id: op.id,
                op_type: OpType::CX,
                qubits: (op.qubits.1, op.qubits.0),
            },
        ],

        // CP things
        (OpType::CP, _) => vec![Operation {
            id: fop.id,
            op_type: OpType::U,
            qubits: (op.qubits.0, op.qubits.1),
        }],
        (_, OpType::CP) => vec![Operation {
            id: fop.id,
            op_type: OpType::U,
            qubits: (op.qubits.0, op.qubits.1),
        }],

        // TODO: CZ with things

        // Default: just output the inputs
        _ => vec![fop.clone(), op.clone()],
    }
}

pub fn match_ops(op: &Operation, fop: &Operation) -> MatchingOps {
    match (op.op_type, fop.op_type) {
        // Cancels
        (OpType::SWAP, OpType::SWAP) => MatchingOps::RemoveFOp,
        (OpType::CZ, OpType::CZ) => MatchingOps::RemoveFOp,
        (OpType::CX, OpType::CX) if op.qubits == fop.qubits => MatchingOps::RemoveFOp,

        // Passthrough
        (OpType::SWAP, OpType::CX) => MatchingOps::SwapPassthrough,
        (OpType::SWAP, OpType::CZ) => MatchingOps::SwapPassthrough,
        (OpType::SWAP, OpType::U) => MatchingOps::SwapPassthrough,
        (OpType::SWAP, OpType::B(_cost, _op_id)) => MatchingOps::SwapPassthrough,

        // Front Op absorbs
        (_, OpType::U) => MatchingOps::FOpAbsorbs,
        //(OpType::U, OpType::U) => MatchingOps::FOpAbsorbs,

        // Remove FOp and apply other op
        (OpType::U, _) => MatchingOps::RemoveFOpAndApply(merge_ops(op, fop)),
        (OpType::CX, OpType::SWAP) => MatchingOps::RemoveFOpAndApply(merge_ops(op, fop)),
        (OpType::CP, _) => MatchingOps::RemoveFOpAndApply(merge_ops(op, fop)),
        (_, OpType::CP) => MatchingOps::RemoveFOpAndApply(merge_ops(op, fop)),

        _ => MatchingOps::JustAddOp,
    }
}
