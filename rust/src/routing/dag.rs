use std::vec;

use super::config::*;

//use indexmap::IndexMap;
use super::graph::TopologicalGenerations;
use super::ops::{match_ops, op_cost, MatchingOps, OpType, Operation};
use petgraph::{
    stable_graph::{NodeIndex, StableDiGraph},
    Direction::Incoming,
};
use rand::{seq::SliceRandom, thread_rng};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Front {
    NoFront,
    DifferentFronts((NodeIndex, NodeIndex)),
    FirstFront(NodeIndex),
    SecondFront(NodeIndex),
    UniqueFront(NodeIndex),
    ClassicalFront((Option<NodeIndex>, Option<NodeIndex>)),
    BarrierFront,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub initial_layout: Vec<usize>,
    pub qubits: Vec<usize>,
    pub locations: Vec<usize>,
}

pub struct TwoQubitDAG {
    pub dag: StableDiGraph<Operation, ()>,
    front: HashMap<usize, NodeIndex>,
    creg_front: Option<NodeIndex>,
    pub layout: Vec<usize>,
    total_gates: usize,
    pub score: usize,
    auto: bool,
    pub virtual_swap: bool,
    gens: Vec<Vec<NodeIndex>>,
    loc_to_act: Vec<Vec<usize>>,
    pub topgens: TopologicalGenerations,
    update_topgens: bool,
}

impl TwoQubitDAG {
    pub fn new(
        auto: bool,
        virtual_swap: bool,
        coupling_map: &Vec<(usize, usize)>,
        num_qubits: usize,
        update_topgens: bool,
    ) -> Self {
        let mut tqd = TwoQubitDAG {
            dag: StableDiGraph::new(),
            front: HashMap::new(),
            creg_front: None,
            layout: (0..num_qubits).collect(),
            total_gates: 0,
            score: 0,
            auto: auto,
            virtual_swap: virtual_swap,
            gens: Vec::new(),
            loc_to_act: Vec::new(),
            topgens: TopologicalGenerations::new(&StableDiGraph::new()),
            update_topgens: update_topgens,
        };

        for _i in 0..num_qubits {
            tqd.loc_to_act.push(Vec::new());
        }

        for (i, (l1, l2)) in coupling_map.iter().enumerate() {
            tqd.loc_to_act[*l1].push(i);
            tqd.loc_to_act[*l2].push(i);
        }

        tqd
    }

    pub fn get_previous_front(&self, idx: &NodeIndex, q: usize) -> Option<NodeIndex> {
        let mut pred_front: Option<NodeIndex> = None;
        //println!("Getting previous front for {q} and node {:?}", idx);
        for pred in self.dag.neighbors_directed(*idx, Incoming) {
            let pred_op = &self.dag[pred];
            let (pq1, pq2) = pred_op.qubits;
            if (q == pq1) || (q == pq2) {
                if pred_front == None {
                    pred_front = Some(pred);
                } else {
                    let pred_front_op = &self.dag[pred_front.unwrap()];
                    if pred_op.id > pred_front_op.id {
                        pred_front = Some(pred);
                    }
                }
            }
        }
        pred_front
    }

    pub fn remove(&mut self, idx: &NodeIndex) {
        let op = &self.dag[*idx];
        let (q1, q2) = op.qubits;
        //println!("removing.. fronts: {:?}", self.front);

        for q in [q1, q2] {
            if let Some(prev_front) = self.get_previous_front(idx, q) {
                self.front.insert(q, prev_front);
            } else {
                self.front.remove(&q);
            }
        }

        self.remove_node(*idx);
        //println!("removed! fronts: {:?}", self.front);
    }

    /// Add a node
    pub fn push(&mut self, op: &Operation) {
        let (a, b) = op.qubits;
        //println!("Current dag: {:?}", self.dag);
        //println!("Current front: {:?}", self.front);
        //println!("Adding {:?}...", op);

        // Insert node and get idx
        let op = Operation {
            id: self.total_gates,
            op_type: op.op_type,
            qubits: op.qubits,
        };
        self.total_gates += 1;

        match self.get_unique_front(&op) {
            Front::DifferentFronts((f1, f2)) => {
                let idx = self.add_node(op);
                self.dag.add_edge(f1, idx, ());
                self.dag.add_edge(f2, idx, ());

                // Insert op into front
                self.front.insert(a, idx);
                self.front.insert(b, idx);
            }
            Front::UniqueFront(f) => {
                let front_op = &self.dag[f].clone();
                //println!("op {:?}", op);
                //println!("fop {:?}", front_op);

                match match_ops(&op, &front_op) {
                    MatchingOps::RemoveFOp if self.auto => {
                        //println!("<RemoveFOp>");
                        self.remove(&f);
                        //println!("Removed FOp!");
                        //println!("Dag: {:?}", self.dag);
                        //println!("</RemoveFOp>");
                    }
                    MatchingOps::SwapPassthrough if self.auto => {
                        //println!("<SwapPassthrough>");
                        self.remove(&f);
                        //println!("Removed FOp!");
                        self.push(&op);
                        //println!("Pushed SWAP!");
                        self.push(&Operation {
                            id: front_op.id,
                            op_type: front_op.op_type,
                            qubits: (front_op.qubits.1, front_op.qubits.0),
                        });
                        //println!("Pushed FOp!");
                        //println!("Dag: {:?}", self.dag);
                        //println!("</SwapPassthrough>");
                    }
                    MatchingOps::FOpAbsorbs if self.auto => {
                        //println!("<FOpAbsorbs>");
                        //println!("FOp absorbed Op, doing nothing!");
                        //println!("</FOpAbsorbs>");
                    }
                    MatchingOps::RemoveFOpAndApply(new_ops) if self.auto => {
                        //println!("<RemoveFOpAndApply>");
                        self.remove(&f);
                        //println!("Removed FOp!");
                        for (_iop, new_op) in new_ops.iter().enumerate() {
                            self.push(new_op);
                            //println!("Pushed an Op");
                        }
                        //println!("</RemoveFOpAndApply>");
                    }
                    _ => {
                        // MatchingOps::JustAddOp
                        let idx = self.add_node(op);
                        self.dag.add_edge(f, idx, ());
                        self.front.insert(a, idx);
                        self.front.insert(b, idx);
                    }
                };
            }
            Front::FirstFront(f1) => {
                let idx = self.add_node(op);
                self.dag.add_edge(f1, idx, ());

                // Insert op into front
                self.front.insert(a, idx);
                self.front.insert(b, idx);
            }
            Front::SecondFront(f2) => {
                let idx = self.add_node(op);
                self.dag.add_edge(f2, idx, ());

                // Insert op into front
                self.front.insert(a, idx);
                self.front.insert(b, idx);
            }
            Front::NoFront => {
                match op.op_type {
                    OpType::SWAP if self.virtual_swap => {
                        // virtual swap
                        let (q1, q2) = op.qubits;
                        (self.layout[q1], self.layout[q2]) = (self.layout[q2], self.layout[q1]);
                        //println!("Doing virtual swap between {q1} and {q2}");
                    }
                    OpType::MRW => {
                        let idx = self.add_node(op);
                        // Insert op into front
                        self.front.insert(a, idx); // using 'a' as qubit since both are equal
                        self.creg_front = Some(idx); // set the classical front
                    }
                    _ => {
                        let idx = self.add_node(op);
                        // Insert op into front
                        self.front.insert(a, idx);
                        self.front.insert(b, idx);
                    }
                }
            }
            Front::ClassicalFront((qf, cf)) => {
                let idx = self.add_node(op);
                if qf.is_some() {
                    self.dag.add_edge(qf.unwrap(), idx, ()); // add edge to the quantum front
                }
                if cf.is_some() && (qf != cf) {
                    self.dag.add_edge(cf.unwrap(), idx, ()); // add edge to the classical front
                }

                // Insert op into front
                self.front.insert(a, idx); // using 'a' as qubit since both are equal
                self.creg_front = Some(idx); // set the classical front
            }
            Front::BarrierFront => {
                let idx = self.add_node(op);

                // Add edges to all the current front unless they are a barrier, where we take the predecessor
                for (&q_i, idx_i) in self.front.iter() {
                    let mut idx_front = Some((*idx_i).clone());

                    // Loop until it is not a barrier or no front is found
                    //while idx_front.is_some()
                    //    && (self.dag.node_weight(idx_front.unwrap()).unwrap().op_type
                    //        == OpType::BARRIER)
                    //{
                    //    idx_front = self.get_previous_front(&idx_front.unwrap(), q_i);
                    //}

                    if idx_front.is_some() && !self.dag.contains_edge(idx_front.unwrap(), idx) {
                        self.dag.add_edge(idx_front.unwrap(), idx, ());
                    }
                }

                // Insert op into front
                self.front.insert(a, idx); // using 'a' as qubit since both are equal
            }
            Front::Error => {
                eprintln!("Warning: DAG Front Error for op {:?}, adding without edges", op);
                let idx = self.add_node(op);
                self.front.insert(a, idx);
                self.front.insert(b, idx);
            }
        };

        //println!("!Added {:?}!", op3);
        //println!("Final dag: {:?}", self.dag);
        //println!("Final front: {:?}", self.front);
    }

    pub fn create_topgens(&mut self) {
        self.topgens = TopologicalGenerations::new(&self.dag);
    }

    pub fn update_gens(&mut self) {
        let mut gens = self.topgens.get_layers(HORIZON, &self.dag);
        for i in 0..gens.len() {
            gens[i].shuffle(&mut thread_rng());
        }
        self.gens = gens;
    }

    pub fn get_active_swaps(&self, locations: &Vec<usize>) -> Vec<usize> {
        let mut active_swaps: Vec<usize> = Vec::with_capacity(NUM_ACTIVE_SWAPS);

        for gen in self.gens.iter() {
            if active_swaps.len() == NUM_ACTIVE_SWAPS {
                break;
            };
            for idx in gen {
                if active_swaps.len() == NUM_ACTIVE_SWAPS {
                    break;
                };
                let (q1, q2) = self.dag[*idx].qubits;
                for q in [q1, q2] {
                    for a in self.loc_to_act[locations[q]].iter() {
                        if active_swaps.len() == NUM_ACTIVE_SWAPS {
                            active_swaps.shuffle(&mut thread_rng());
                            return active_swaps;
                        };
                        if !active_swaps.contains(a) {
                            active_swaps.push(*a);
                        };
                    }
                }
            }
        }

        active_swaps.shuffle(&mut thread_rng());
        active_swaps
    }

    /// Get the obs for the model based on current dag
    pub fn get_obs(
        &self,
        locations: &Vec<usize>,
        coupling_map: &Vec<(usize, usize)>,
        dists: &Vec<Vec<DistType>>,
    ) -> (Vec<f32>, Vec<usize>) {
        let mut out: Vec<f32> = vec![0.0; NUM_ACTIVE_SWAPS * HORIZON];
        //-//println!("{:?}", self.dag);

        ////println!("{:?}", gens);
        let active_swaps = self.get_active_swaps(locations);

        for (d, gen) in self.gens.iter().enumerate() {
            for isw in 0..active_swaps.len() {
                let sw = active_swaps[isw];
                let (a, b) = coupling_map[sw];
                for idx in gen {
                    let (aq, bq) = self.dag[*idx].qubits;
                    let (al, bl) = (locations[aq], locations[bq]);
                    if (al == a) && (bl != b) {
                        out[d + isw * HORIZON] += (dists[a][bl] - dists[b][bl]) as f32;
                    } else if (al == b) && (bl != a) {
                        out[d + isw * HORIZON] += (dists[b][bl] - dists[a][bl]) as f32;
                    } else if (bl == a) && (al != b) {
                        out[d + isw * HORIZON] += (dists[a][al] - dists[b][al]) as f32;
                    } else if (bl == b) && (al != a) {
                        out[d + isw * HORIZON] += (dists[b][al] - dists[a][al]) as f32;
                    }
                }
            }
        }
        (out, active_swaps)
    }

    /// Number of nodes in the set.
    pub fn len(&self) -> usize {
        self.dag.node_count()
    }

    pub fn add_node(&mut self, op: Operation) -> NodeIndex {
        self.score += op_cost(op.op_type);
        self.dag.add_node(op)
    }

    pub fn remove_node(&mut self, idx: NodeIndex) {
        if self.update_topgens {
            self.topgens.pop(idx, &self.dag);
        }
        let op = self.dag.remove_node(idx).unwrap();
        self.score -= op_cost(op.op_type);
    }

    pub fn get_unique_front(&self, op: &Operation) -> Front {
        let (a, b) = op.qubits;
        //return Front::DifferentFronts;

        match op.op_type {
            OpType::BARRIER => Front::BarrierFront,
            OpType::MRW => Front::ClassicalFront((self.front.get(&a).copied(), self.creg_front)),
            _ => {
                match (self.front.get(&a), self.front.get(&b)) {
                    (None, None) => Front::NoFront,
                    (Some(&value_a), Some(&value_b)) if value_a == value_b => {
                        Front::UniqueFront(value_a)
                    }
                    (Some(&value_a), Some(&value_b)) if value_a != value_b => {
                        Front::DifferentFronts((value_a, value_b))
                    }
                    (Some(&value_a), None) => Front::FirstFront(value_a),
                    (None, Some(&value_b)) => Front::SecondFront(value_b),
                    _ => {
                        //println!("ERROR");
                        Front::Error
                    }
                }
            }
        }
    }
}
