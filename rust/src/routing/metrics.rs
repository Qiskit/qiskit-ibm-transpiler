use super::{ops::OpType, ops::Operation};
use std::collections::{HashMap, HashSet};

pub type MetricType = Vec<usize>;

pub struct CircuitMetrics {
    n_cnots: usize,
    n_gates: usize,
    cnot_layers: HashSet<usize>,
    layers: HashSet<usize>,
    last_gates: Vec<isize>,
    last_cxs: Vec<isize>,
    fid: f64,
    err_map: HashMap<(usize, usize), f64>,
    metrics_names: Vec<String>,
}

impl CircuitMetrics {
    pub fn new(
        num_qubits: usize,
        metrics_names: &[String],
        err_map: &HashMap<(usize, usize), f64>,
    ) -> Self {
        CircuitMetrics {
            n_cnots: 0,
            n_gates: 0,
            cnot_layers: HashSet::new(),
            layers: HashSet::new(),
            last_gates: vec![-1; num_qubits],
            last_cxs: vec![-1; num_qubits],
            fid: 1.0,
            err_map: err_map.clone(),
            metrics_names: metrics_names.to_vec(),
        }
    }

    pub fn cx(&mut self, q0: usize, q1: usize) {
        self.n_cnots += 1;
        self.n_gates += 1;

        // TODO: Handle wubits with no noise
        if self.err_map.contains_key(&(q0, q1)) {
            self.fid *= 1.0 - self.err_map[&(q0, q1)];
        }

        let gate_layer = 1 + std::cmp::max(self.last_gates[q0], self.last_gates[q1]);
        let cx_layer = 1 + std::cmp::max(self.last_cxs[q0], self.last_cxs[q1]);

        self.last_gates[q0] = gate_layer as isize;
        self.last_gates[q1] = gate_layer as isize;
        self.last_cxs[q0] = cx_layer as isize;
        self.last_cxs[q1] = cx_layer as isize;

        self.cnot_layers.insert(cx_layer as usize);
        self.layers.insert(gate_layer as usize);
    }

    pub fn gate(&mut self, q: usize) {
        self.n_gates += 1;
        let gate_layer = self.last_gates[q] + 1;
        self.last_gates[q] = gate_layer;
        self.layers.insert(gate_layer as usize);
    }

    pub fn apply(&mut self, op: &Operation) {
        match op.op_type {
            OpType::CX | OpType::CZ => self.cx(op.qubits.0, op.qubits.1),
            OpType::SWAP | OpType::U => {
                self.cx(op.qubits.0, op.qubits.1);
                self.cx(op.qubits.0, op.qubits.1);
                self.cx(op.qubits.0, op.qubits.1);
            }
            OpType::CP => {
                self.cx(op.qubits.0, op.qubits.1);
                self.cx(op.qubits.0, op.qubits.1);
            }
            OpType::B(cost, _op_id) => {
                for _ in 0..cost {
                    self.cx(op.qubits.0, op.qubits.1);
                }
            }
            OpType::MRW => {}
            OpType::BARRIER => {}
        }
    }

    fn get_metric_by_name(&self, metric_name: &str) -> usize {
        match metric_name.to_lowercase().as_str() {
            "noise" => ((1. - self.fid) * 1_000_000.) as usize, // To keep things simple we convert this to usize as it is a value between 0 & 1
            "cnot_layers" => self.cnot_layers.len(),
            "n_cnots" => self.n_cnots,
            "layers" => self.layers.len(),
            "n_gates" => self.n_gates,
            unknown => {
                eprintln!("Warning: unknown metric name '{}', returning 0", unknown);
                0
            }
        }
    }

    pub fn get_metrics(&self) -> MetricType {
        let mut metrics = Vec::with_capacity(self.metrics_names.len());
        for metric_name in self.metrics_names.iter() {
            metrics.push(self.get_metric_by_name(metric_name));
        }
        metrics
    }
}

pub fn metrics_from_circuit(
    circuit: &[Operation],
    num_qubits: usize,
    metrics_names: &[String],
    err_map: &HashMap<(usize, usize), f64>,
) -> MetricType {
    let mut cm = CircuitMetrics::new(num_qubits, metrics_names, err_map);
    for op in circuit {
        cm.apply(op);
    }
    cm.get_metrics()
}
