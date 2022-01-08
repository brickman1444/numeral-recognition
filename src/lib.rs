use wasm_bindgen::prelude::*;

pub const PIXELS_PER_IMAGE: usize = 28 * 28;

#[wasm_bindgen]
pub fn recognize(neural_network_json_text: &str, image_bytes: &[u8]) -> u8 {
    std::println!("Load neural network from json");
    let neural_network = nn::NN::from_json(neural_network_json_text);

    let mut input_vec = std::vec::Vec::with_capacity(PIXELS_PER_IMAGE);
    for item in image_bytes.iter().take(PIXELS_PER_IMAGE) {
        input_vec.push((*item as f64) / 255f64);
    }

    let result_vector = neural_network.run(input_vec.as_slice());

    evaluate_label_vector(&result_vector)
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
