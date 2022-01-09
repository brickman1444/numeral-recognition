use wasm_bindgen::prelude::*;

pub const PIXELS_PER_IMAGE: usize = 28 * 28;

#[wasm_bindgen]
pub struct RecognitionResults {
    pub first_guess: u8,
    pub first_guess_confidence: f64,
    pub second_guess: u8,
    pub second_guess_confidence: f64,
}

#[wasm_bindgen]
pub fn recognize(neural_network_json_text: &str, image_bytes: &[u8]) -> RecognitionResults {
    println!("Load neural network from json");
    let neural_network = nn::NN::from_json(neural_network_json_text);

    let input_vec: Vec<f64> = image_bytes
        .iter()
        .take(PIXELS_PER_IMAGE)
        .map(|item| (*item as f64) / 255f64)
        .collect();

    let result_vector = neural_network.run(input_vec.as_slice());

    let first_guess = evaluate_label_vector(&result_vector);

    RecognitionResults {
        first_guess: first_guess,
        first_guess_confidence: 0.5f64,
        second_guess: 10,
        second_guess_confidence: 0.2f64,
    }
}

pub fn evaluate_label_vector(label_vec: &[f64]) -> u8 {
    let (index_of_max_value, _) = // Second value is max value
        label_vec.iter()
            .enumerate()
            .fold((0, label_vec[0]), |(idx_max, val_max), (idx, val)| {
                if &val_max > val {
                    (idx_max, val_max)
                } else {
                    (idx, *val)
                }
            });
    index_of_max_value as u8
}
