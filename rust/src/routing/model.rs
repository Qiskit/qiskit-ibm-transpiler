use super::model_data::ModelData;
use nalgebra::SVector;

fn relu(v: f32) -> f32 {
    v.max(0.0)
}

pub fn forward(model: &ModelData, s: &[f32]) -> SVector<f32, 16> {
    let mut acts = model.bias0.clone();

    for (i, &v) in s.iter().enumerate() {
        if v > 0.0 {
            acts += model.embeddings[i];
        } else if v < 0.0 {
            acts -= model.embeddings[i];
        }
    }

    let acts = model.layer1 * acts.map(relu) + model.bias1;
    acts
}

pub fn predict(model: &ModelData, s: &[f32]) -> usize {
    forward(model, s).imax() as usize
}
