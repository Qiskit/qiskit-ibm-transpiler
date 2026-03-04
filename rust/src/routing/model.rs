use super::model_data::{BIAS0, BIAS1, EMBEDDINGS, LAYER1};
use nalgebra::SVector;

pub struct Model;

use rand::Rng;

fn relu(v: f32) -> f32 {
    v.max(0.0)
}

fn noise(v: f32) -> f32 {
    // see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    let r: f32 = rand::thread_rng().gen::<f32>();
    v - (-(r.ln())).ln()
}

pub fn forward(s: &Vec<f32>) -> SVector<f32, 16> {
    let mut acts = BIAS0.clone();

    for (i, &v) in s.iter().enumerate() {
        if v > 0.0 {
            acts += EMBEDDINGS[i];
        } else if v < 0.0 {
            acts -= EMBEDDINGS[i];
        }
    }

    let acts = LAYER1 * acts.map(relu) + BIAS1;
    acts
}

pub fn predict(s: &Vec<f32>) -> usize {
    forward(s).imax() as usize
}

pub fn predict_sample(s: &Vec<f32>) -> usize {
    forward(s).map(noise).imax() as usize
}
