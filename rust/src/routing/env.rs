use super::config::*;
use super::dag::{Layout, TwoQubitDAG};
use super::metrics::{CircuitMetrics, MetricType};
use super::ops::{OpType, Operation};
use petgraph::{algo, prelude::*};

use std::collections::HashMap;
use std::collections::VecDeque;

pub struct Env {
    pub in_dag: TwoQubitDAG,
    pub out_dag: TwoQubitDAG,
    pub active_swaps: Vec<usize>,
    qubits: Vec<usize>,
    locations: Vec<usize>,
    cm: CircuitMetrics,
}

impl Env {
    pub fn new(
        input_circuit: &[Operation],
        coupling_map: &[(usize, usize)],
        err_map: &HashMap<(usize, usize), f64>,
        metrics_names: &[String],
        num_qubits: usize,
    ) -> Self {
        let mut env = Env {
            in_dag: TwoQubitDAG::new(false, false, coupling_map, num_qubits, true),
            out_dag: TwoQubitDAG::new(true, true, coupling_map, num_qubits, false),
            active_swaps: (0..NUM_ACTIVE_SWAPS).collect(),
            qubits: (0..num_qubits).collect(),
            locations: (0..num_qubits).collect(),
            cm: CircuitMetrics::new(num_qubits, metrics_names, err_map),
        };

        for op in input_circuit {
            env.in_dag.push(op);
        }
        env.in_dag.create_topgens();

        env
    }

    pub fn execute_operations(&mut self, dists: &[Vec<DistType>]) -> usize {
        let mut num_targets_solved = 0usize;
        let mut roots: VecDeque<NodeIndex> = self.in_dag.topgens.zero_indegree.clone();

        while let Some(node) = roots.pop_front() {
            let op = self.in_dag.dag.node_weight(node).unwrap().clone();
            let (q1, q2) = op.qubits;
            let l1 = self.locations[q1];
            let l2 = self.locations[q2];

            if dists[l1][l2] <= 1 {
                num_targets_solved += 1;

                let new_op = Operation {
                    id: self.out_dag.len(),
                    op_type: op.op_type,
                    qubits: (l1, l2),
                };
                self.out_dag.push(&new_op);
                self.cm.apply(&new_op);

                // Remove node and process successors
                let successors: Vec<_> = self.in_dag.dag.neighbors(node).collect();
                self.in_dag.remove_node(node);

                for successor in successors {
                    if self
                        .in_dag
                        .dag
                        .neighbors_directed(successor, Incoming)
                        .count()
                        == 0
                    {
                        roots.push_back(successor);
                    }
                }
            }
        }

        num_targets_solved
    }

    pub fn swap(&mut self, action: usize, coupling_map: &[(usize, usize)]) {
        if action >= self.active_swaps.len() {
            return;
        }
        let action = self.active_swaps[action];
        let (l1, l2) = coupling_map[action];
        let op = Operation {
            id: self.out_dag.len(),
            op_type: OpType::SWAP,
            qubits: (l1, l2),
        };

        self.out_dag.push(&op);
        self.cm.apply(&op);

        // Swap locations
        (self.qubits[l1], self.qubits[l2]) = (self.qubits[l2], self.qubits[l1]);
        let (q1, q2) = (self.qubits[l1], self.qubits[l2]);
        (self.locations[q1], self.locations[q2]) = (l1, l2);
    }

    pub fn routed(&self) -> bool {
        self.in_dag.len() == 0
    }

    pub fn obs(
        &mut self,
        coupling_map: &[(usize, usize)],
        dists: &[Vec<DistType>],
    ) -> Vec<f32> {
        self.in_dag.update_gens();
        let (obs, active_swaps) = self.in_dag.get_obs(&self.locations, coupling_map, dists);
        self.active_swaps = active_swaps;
        obs
    }

    pub fn get_circuit(&self) -> Vec<Operation> {
        let mut out: Vec<Operation> = Vec::with_capacity(self.out_dag.len());
        if !self.routed() {
            return out;
        }
        match algo::toposort(&self.out_dag.dag, None) {
            Ok(order) => {
                for idx in &order {
                    let op = self.out_dag.dag[*idx].clone();
                    out.push(op);
                }
            }
            Err(_e) => {

            }
        }
        out
    }

    pub fn get_score(&self) -> MetricType {
        if self.routed() {
            self.cm.get_metrics()
        } else {
            vec![usize::MAX]
        }
    }

    pub fn get_initial_layout(&self) -> &Vec<usize> {
        &self.out_dag.layout
    }
    pub fn get_qubits(&self) -> &Vec<usize> {
        &self.qubits
    }
    pub fn get_locations(&self) -> &Vec<usize> {
        &self.locations
    }

    pub fn get_layout(&self) -> Layout {
        Layout {
            initial_layout: self.out_dag.layout.clone(),
            qubits: self.qubits.clone(),
            locations: self.locations.clone(),
        }
    }
}
