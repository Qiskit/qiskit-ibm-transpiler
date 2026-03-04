use super::ops::Operation;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::{HashMap, VecDeque};

pub struct TopologicalGenerations {
    indegree_map: HashMap<NodeIndex, usize>,
    pub zero_indegree: VecDeque<NodeIndex>,
}

impl TopologicalGenerations {
    pub fn new(graph: &StableDiGraph<Operation, ()>) -> Self {
        let mut indegree_map = HashMap::new();
        let mut zero_indegree = VecDeque::new();

        for node in graph.node_identifiers() {
            let indegree = graph
                .edges_directed(node, petgraph::Direction::Incoming)
                .count();
            if indegree == 0 {
                zero_indegree.push_back(node);
            } else {
                indegree_map.insert(node, indegree);
            }
        }

        TopologicalGenerations {
            indegree_map,
            zero_indegree,
        }
    }

    // Pop a node from the graph and update the indegree_map and zero_indegree
    pub fn pop(&mut self, node: NodeIndex, graph: &StableDiGraph<Operation, ()>) {
        // Remove the node from the zero_indegree queue if it is there
        if let Some(pos) = self.zero_indegree.iter().position(|&n| n == node) {
            self.zero_indegree.remove(pos);
        }

        // For each neighbor of the node, reduce its indegree
        for neighbor in graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
            if let Some(indegree) = self.indegree_map.get_mut(&neighbor) {
                *indegree -= 1;
                if *indegree == 0 {
                    // Move to zero_indegree if it has no other incoming edges
                    self.indegree_map.remove(&neighbor);
                    self.zero_indegree.push_back(neighbor);
                }
            }
        }
    }

    pub fn get_layers(
        &mut self,
        num_layers: usize,
        graph: &StableDiGraph<Operation, ()>,
    ) -> Vec<Vec<NodeIndex>> {
        let mut layers = Vec::new();
        let original_zero_indegree = self.zero_indegree.clone(); // Clone the small zero_indegree queue
        let mut indegree_changes = Vec::new();

        for _ in 0..num_layers {
            if self.zero_indegree.is_empty() {
                break;
            }

            let this_generation = std::mem::take(&mut self.zero_indegree);
            let mut next_generation = VecDeque::new();

            for &node in &this_generation {
                for neighbor in graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                    let entry = self.indegree_map.entry(neighbor).or_insert(0);
                    indegree_changes.push((neighbor, *entry)); // Track the original indegree
                    *entry -= 1;

                    if *entry == 0 {
                        self.indegree_map.remove(&neighbor);
                        next_generation.push_back(neighbor);
                    }
                }
            }

            layers.push(this_generation.into_iter().collect());

            // Move the next generation to zero_indegree for the next iteration
            self.zero_indegree = next_generation;
        }

        // Revert the indegree changes
        for (node, original_indegree) in indegree_changes.into_iter().rev() {
            *self.indegree_map.entry(node).or_insert(0) = original_indegree;
        }

        // Restore the original zero_indegree queue
        self.zero_indegree = original_zero_indegree;

        layers
    }
}
