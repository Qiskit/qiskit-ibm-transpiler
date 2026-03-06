use super::model_data::ModelData;
use nalgebra::DVector;

pub fn forward(model: &ModelData, s: &[f32]) -> DVector<f32> {
    let mut acts = model.bias0.clone();

    for (i, &v) in s.iter().enumerate() {
        if v > 0.0 {
            acts += &model.embeddings[i];
        } else if v < 0.0 {
            acts -= &model.embeddings[i];
        }
    }

    &model.layer1 * &acts.map(|v| v.max(0.0)) + &model.bias1
}

pub fn predict(model: &ModelData, s: &[f32]) -> usize {
    forward(model, s).imax()
}
