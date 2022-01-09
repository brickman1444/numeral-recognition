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

    evaluate_label_vector(&result_vector)
}

pub fn evaluate_label_vector(label_vec: &[f64]) -> RecognitionResults {
    let mut indices_and_confidence: Vec<(usize, &f64)> = label_vec.iter().enumerate().collect();

    indices_and_confidence.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    RecognitionResults {
        first_guess: indices_and_confidence[0].0 as u8,
        first_guess_confidence: *indices_and_confidence[0].1,
        second_guess: indices_and_confidence[1].0 as u8,
        second_guess_confidence: *indices_and_confidence[1].1,
    }
}
