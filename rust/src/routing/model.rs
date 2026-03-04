use super::model_data::ModelData;
use nalgebra::SVector;

use rand::Rng;

fn relu(v: f32) -> f32 {
    v.max(0.0)
}

fn noise(v: f32) -> f32 {
    // see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    let r: f32 = rand::thread_rng().gen::<f32>();
    v - (-(r.ln())).ln()
}

pub fn forward(model: &ModelData, s: &Vec<f32>) -> SVector<f32, 16> {
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

pub fn predict(model: &ModelData, s: &Vec<f32>) -> usize {
    forward(model, s).imax() as usize
}

pub fn predict_sample(model: &ModelData, s: &Vec<f32>) -> usize {
    forward(model, s).map(noise).imax() as usize
}
