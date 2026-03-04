use super::config::DistType;
use super::dag::Layout;
use super::env::Env;
use super::metrics::{metrics_from_circuit, MetricType};
use super::model::predict;
use super::ops::{optype_to_usize, OpType, Operation, OP_TYPES};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::panic;
use std::thread;
use std::time::{Duration, Instant};

pub const MAX_STEPS: usize = 200000000;

#[pyclass]
pub struct CircuitRouting {
    pub routing: Routing,
    pub transpiling: Routing,
}
#[pymethods]
impl CircuitRouting {
    #[new]
    pub fn new() -> Self {
        CircuitRouting {
            routing: Routing::new(false),
            transpiling: Routing::new(true),
        }
    }

    #[staticmethod]
    pub fn get_dists_max_value() -> DistType {
        return DistType::MAX;
    }

    pub fn transpile(
        &self,
        circuit: Vec<(usize, (usize, usize))>,
        runs: usize,
        inner_its: usize,
        its: usize,
        shots: usize,
        layout: Vec<Vec<usize>>,
        coupling_map: Vec<(usize, usize)>,
        dists: Vec<Vec<i32>>,
        err_map: HashMap<(usize, usize), f64>,
        metrics_names: Vec<String>,
        num_qubits: usize,
        max_seconds: usize,
    ) -> PyResult<(
        Vec<(usize, (usize, usize))>,
        (Vec<usize>, Vec<usize>, Vec<usize>),
    )> {
        let route_result = panic::catch_unwind(|| {
            self.transpiling.route(
                &py_to_rust(circuit),
                runs,
                inner_its,
                its,
                shots,
                layout,
                coupling_map,
                dists,
                err_map,
                metrics_names,
                num_qubits,
                max_seconds,
            )
        });
        match route_result {
            Err(_e) => Err(PyRuntimeError::new_err("Panic error")),
            Ok(res) => Ok((
                rust_to_py(res.0),
                (res.1.initial_layout, res.1.qubits, res.1.locations),
            )),
        }
    }

    pub fn route(
        &self,
        circuit: Vec<(usize, (usize, usize))>,
        runs: usize,
        coupling_map: Vec<(usize, usize)>,
        dists: Vec<Vec<i32>>,
        err_map: HashMap<(usize, usize), f64>,
        metrics_names: Vec<String>,
        num_qubits: usize,
    ) -> (
        Vec<(usize, (usize, usize))>,
        (Vec<usize>, Vec<usize>, Vec<usize>),
    ) {
        let route_result = self.routing._route_loop(
            &py_to_rust(circuit),
            runs,
            &coupling_map,
            &dists,
            &err_map,
            &metrics_names,
            num_qubits,
        );
        (
            rust_to_py(route_result.0),
            (
                route_result.1.initial_layout,
                route_result.1.qubits,
                route_result.1.locations,
            ),
        )
    }
}

fn py_to_rust(circuit: Vec<(usize, (usize, usize))>) -> Vec<Operation> {
    circuit
        .iter()
        .enumerate()
        .map(|(op_id, (gate_id, qubits))| Operation {
            id: op_id,
            op_type: {
                if *gate_id <= 9 {
                    OP_TYPES[*gate_id]
                } else {
                    OpType::B(3, *gate_id)
                }
            },
            qubits: *qubits,
        })
        .collect()
}

fn rust_to_py(circuit: Vec<Operation>) -> Vec<(usize, (usize, usize))> {
    circuit
        .iter()
        .map(|op| (optype_to_usize(op.op_type), op.qubits))
        .collect()
}

fn apply_layout(circuit: &mut Vec<Operation>, layout: &Vec<usize>) {
    for op in circuit.iter_mut() {
        op.qubits = (layout[op.qubits.0], layout[op.qubits.1])
    }
}

fn compose_layout(base_layout: &Vec<usize>, new_layout: &Vec<usize>) -> Vec<usize> {
    base_layout.iter().map(|q| new_layout[*q]).collect()
}

fn invert_layout(layout: &mut Layout) {
    (layout.initial_layout, layout.locations) =
        //(argsort(&layout.locations), argsort(&layout.initial_layout))
        (layout.locations.clone(), argsort(&layout.initial_layout))
}

pub fn argsort(data: &Vec<usize>) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

pub struct Routing {
    pub virtual_swap: bool,
}

impl Routing {
    pub fn new(virtual_swap: bool) -> Self {
        Routing {
            virtual_swap: virtual_swap,
        }
    }

    fn _route(
        &self,
        circuit: &Vec<Operation>,
        coupling_map: &Vec<(usize, usize)>,
        dists: &Vec<Vec<i32>>,
        err_map: &HashMap<(usize, usize), f64>,
        metrics_names: &Vec<String>,
        num_qubits: usize,
    ) -> (Vec<Operation>, Layout, MetricType) {
        let mut env = Env::new(circuit, coupling_map, err_map, metrics_names, num_qubits);
        env.out_dag.virtual_swap = self.virtual_swap;

        let mut i: usize = 0usize;
        env.execute_operations(&dists);

        // Choose gates step by step until solved
        while !env.routed() && i < MAX_STEPS {
            let action = predict(&env.obs(&coupling_map, &dists));
            env.swap(action, &coupling_map);
            env.execute_operations(&dists);

            i += 1;
        }
        //(env.get_circuit(), env.get_layout(), env.get_score())
        let qc = env.get_circuit();
        let ms = metrics_from_circuit(&qc, num_qubits, metrics_names, err_map);
        (qc, env.get_layout(), ms)
    }

    fn _route_loop(
        &self,
        circuit: &Vec<Operation>,
        runs: usize,
        coupling_map: &Vec<(usize, usize)>,
        dists: &Vec<Vec<i32>>,
        err_map: &HashMap<(usize, usize), f64>,
        metrics_names: &Vec<String>,
        num_qubits: usize,
    ) -> (Vec<Operation>, Layout, MetricType) {
        let mut best_score: MetricType = vec![usize::MAX];
        let mut best_qc: Vec<Operation> = Vec::new();
        let mut best_layout = Layout {
            initial_layout: (0..num_qubits).collect(),
            qubits: (0..num_qubits).collect(),
            locations: (0..num_qubits).collect(),
        };

        thread::scope(|s| {
            let mut handles = Vec::with_capacity(runs);
            for _ in 0..runs {
                // let value_ref = Arc::clone(&value_ref);
                let handle = s.spawn(|| {
                    self._route(
                        circuit,
                        &coupling_map,
                        &dists,
                        &err_map,
                        &metrics_names,
                        num_qubits,
                    )
                });
                handles.push(handle);
            }
            for handle in handles {
                let (qc, layout, score) = handle.join().expect("Thread panicked");
                if (qc.len() > 0) && (score < best_score) {
                    //println!("    -[{i}] -> Circuit with score {score} found.");
                    (best_qc, best_layout, best_score) = (qc, layout, score);
                }
            }
        });

        (best_qc, best_layout, best_score)
    }

    // Route bidir v1
    pub fn route(
        &self,
        circuit: &Vec<Operation>,
        runs: usize,
        inner_its: usize,
        its: usize,
        shots: usize,
        layout: Vec<Vec<usize>>,
        coupling_map: Vec<(usize, usize)>,
        dists: Vec<Vec<i32>>,
        err_map: HashMap<(usize, usize), f64>,
        metrics_names: Vec<String>,
        num_qubits: usize,
        max_seconds: usize,
    ) -> (Vec<Operation>, Layout, MetricType) {
        let mut best_score: MetricType = vec![usize::MAX];
        let mut best_qc: Vec<Operation> = Vec::new();

        let mut best_layout = Layout {
            initial_layout: (0..num_qubits).collect(),
            qubits: (0..num_qubits).collect(),
            locations: (0..num_qubits).collect(),
        };

        let time_limit = Duration::new(max_seconds as u64, 0);
        let start_time = Instant::now();

        for l in 0..shots {
            let shot_layout: &Vec<usize> = &layout[l];
            let mut best_layout_ci: Vec<usize> = (0..num_qubits).collect();
            let mut best_ci = circuit.clone();

            apply_layout(&mut best_ci, &shot_layout);
            best_layout_ci = compose_layout(&best_layout_ci, &shot_layout);

            for _k in 0..its {
                let mut circuit_i = best_ci.clone();
                let mut layout_i = best_layout_ci.clone();

                for j in 0..inner_its {
                    if j % 2 == 1 {
                        circuit_i.reverse();
                    }

                    let (mut qc, mut layout, score) = self._route_loop(
                        &circuit_i,
                        runs,
                        &coupling_map,
                        &dists,
                        &err_map,
                        &metrics_names,
                        num_qubits,
                    );
                    if j % 2 == 1 {
                        qc.reverse();
                        circuit_i.reverse();
                        invert_layout(&mut layout);
                    }
                    //println!("Score: {:?} vs Best Score {:?}", score, best_score);
                    if score < best_score {
                        //println!(
                        //    "-[{l}/{_k}/{j}]({}s) -> Circuit with metrics: {:?}  found.",
                        //    start_time.elapsed().as_secs(),
                        //    score
                        //);
                        (best_qc, best_layout, best_score) = (qc.clone(), layout.clone(), score);
                        if j == 0 {
                            best_layout.initial_layout = argsort(&best_layout.initial_layout);
                        }
                        best_layout.initial_layout =
                            compose_layout(&layout_i, &best_layout.initial_layout);
                        best_layout.initial_layout = argsort(&best_layout.initial_layout);
                        best_layout.locations = compose_layout(&layout_i, &best_layout.locations);
                        best_ci = circuit_i.clone();
                        best_layout_ci = layout_i.clone();
                    }

                    let new_layout = argsort(&layout.qubits);
                    apply_layout(&mut circuit_i, &new_layout);
                    layout_i = compose_layout(&layout_i, &new_layout);

                    if start_time.elapsed() > time_limit {
                        // Directly return the best result until now if we exceed the time limit
                        return (best_qc, best_layout, best_score);
                    }
                }
            }
        }
        (best_qc, best_layout, best_score)
    }
}
