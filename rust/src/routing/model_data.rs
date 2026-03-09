use super::config::{HORIZON, NUM_ACTIVE_SWAPS};
use nalgebra::{DMatrix, DVector};
use safetensors::SafeTensors;
use std::fs;

#[derive(Clone)]
pub struct ModelData {
    pub bias0: DVector<f32>,
    pub bias1: DVector<f32>,
    pub embeddings: Vec<DVector<f32>>,
    pub layer1: DMatrix<f32>,
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "Byte length must be multiple of 4, got {}",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

impl ModelData {
    pub fn load(path: &str) -> Result<Self, String> {
        let data =
            fs::read(path).map_err(|e| format!("Failed to read model file '{}': {}", path, e))?;
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| format!("Failed to parse safetensors: {}", e))?;

        // Load bias0 and infer hidden_dim from it
        let bias0 = Self::load_vector(&tensors, "bias0")?;
        let hidden_dim = bias0.len();

        // Load bias1 and infer output_dim
        let bias1 = Self::load_vector(&tensors, "bias1")?;
        let output_dim = bias1.len();

        // Validate output_dim matches action space
        if output_dim != NUM_ACTIVE_SWAPS {
            return Err(format!(
                "Model output_dim ({}) != NUM_ACTIVE_SWAPS ({})",
                output_dim, NUM_ACTIVE_SWAPS
            ));
        }

        // Load layer1 and validate shape
        let layer1 = Self::load_matrix(&tensors, "layer1", output_dim, hidden_dim)?;

        // Load embeddings and validate
        let expected_num_embeddings = NUM_ACTIVE_SWAPS * HORIZON;
        let embeddings = Self::load_embeddings(&tensors, expected_num_embeddings, hidden_dim)?;

        Ok(ModelData {
            bias0,
            bias1,
            embeddings,
            layer1,
        })
    }

    fn load_vector(tensors: &SafeTensors, name: &str) -> Result<DVector<f32>, String> {
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("Tensor '{}' not found: {}", name, e))?;
        let floats = bytes_to_f32_vec(view.data())?;
        Ok(DVector::from_column_slice(&floats))
    }

    fn load_matrix(
        tensors: &SafeTensors,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<DMatrix<f32>, String> {
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("Tensor '{}' not found: {}", name, e))?;
        let floats = bytes_to_f32_vec(view.data())?;
        if floats.len() != rows * cols {
            return Err(format!(
                "Tensor '{}' has {} elements, expected {}",
                name,
                floats.len(),
                rows * cols
            ));
        }
        // safetensors stores in row-major (C order), nalgebra expects column-major
        let mut mat = DMatrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                mat[(r, c)] = floats[r * cols + c];
            }
        }
        Ok(mat)
    }

    fn load_embeddings(
        tensors: &SafeTensors,
        num_embeddings: usize,
        hidden_dim: usize,
    ) -> Result<Vec<DVector<f32>>, String> {
        let view = tensors
            .tensor("embeddings")
            .map_err(|e| format!("Tensor 'embeddings' not found: {}", e))?;
        let floats = bytes_to_f32_vec(view.data())?;
        if floats.len() != num_embeddings * hidden_dim {
            return Err(format!(
                "Tensor 'embeddings' has {} elements, expected {} ({}×{})",
                floats.len(),
                num_embeddings * hidden_dim,
                num_embeddings,
                hidden_dim
            ));
        }
        let mut embeddings = Vec::with_capacity(num_embeddings);
        for i in 0..num_embeddings {
            let start = i * hidden_dim;
            embeddings.push(DVector::from_column_slice(
                &floats[start..start + hidden_dim],
            ));
        }
        Ok(embeddings)
    }
}
