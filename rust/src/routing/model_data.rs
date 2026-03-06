use nalgebra::{SMatrix, SVector};
use safetensors::SafeTensors;
use std::fs;

#[derive(Clone)]
pub struct ModelData {
    pub bias0: SVector<f32, 256>,
    pub bias1: SVector<f32, 16>,
    pub embeddings: [SVector<f32, 256>; 128],
    pub layer1: SMatrix<f32, 16, 256>,
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0, "Byte length must be multiple of 4");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl ModelData {
    pub fn load(path: &str) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("Failed to read model file '{}': {}", path, e))?;
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| format!("Failed to parse safetensors: {}", e))?;

        let bias0 = Self::load_vector::<256>(&tensors, "bias0")?;
        let bias1 = Self::load_vector::<16>(&tensors, "bias1")?;
        let layer1 = Self::load_matrix::<16, 256>(&tensors, "layer1")?;
        let embeddings = Self::load_embeddings(&tensors)?;

        Ok(ModelData {
            bias0,
            bias1,
            embeddings,
            layer1,
        })
    }

    fn load_vector<const N: usize>(
        tensors: &SafeTensors,
        name: &str,
    ) -> Result<SVector<f32, N>, String> {
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("Tensor '{}' not found: {}", name, e))?;
        let floats = bytes_to_f32_vec(view.data());
        if floats.len() != N {
            return Err(format!(
                "Tensor '{}' has {} elements, expected {}",
                name,
                floats.len(),
                N
            ));
        }
        Ok(SVector::<f32, N>::from_column_slice(&floats))
    }

    fn load_matrix<const R: usize, const C: usize>(
        tensors: &SafeTensors,
        name: &str,
    ) -> Result<SMatrix<f32, R, C>, String> {
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("Tensor '{}' not found: {}", name, e))?;
        let floats = bytes_to_f32_vec(view.data());
        if floats.len() != R * C {
            return Err(format!(
                "Tensor '{}' has {} elements, expected {}",
                name,
                floats.len(),
                R * C
            ));
        }
        // safetensors stores in row-major (C order), nalgebra expects column-major
        // We need to transpose: read as (R, C) row-major, store as column-major
        let mut mat = SMatrix::<f32, R, C>::zeros();
        for r in 0..R {
            for c in 0..C {
                mat[(r, c)] = floats[r * C + c];
            }
        }
        Ok(mat)
    }

    fn load_embeddings(tensors: &SafeTensors) -> Result<[SVector<f32, 256>; 128], String> {
        let view = tensors
            .tensor("embeddings")
            .map_err(|e| format!("Tensor 'embeddings' not found: {}", e))?;
        let floats = bytes_to_f32_vec(view.data());
        if floats.len() != 128 * 256 {
            return Err(format!(
                "Tensor 'embeddings' has {} elements, expected {}",
                floats.len(),
                128 * 256
            ));
        }
        let mut embeddings: [SVector<f32, 256>; 128] =
            [SVector::<f32, 256>::zeros(); 128];
        for i in 0..128 {
            let start = i * 256;
            embeddings[i] = SVector::<f32, 256>::from_column_slice(&floats[start..start + 256]);
        }
        Ok(embeddings)
    }
}
