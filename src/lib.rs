use std::io::prelude::*;
use wasm_bindgen::prelude::*;

const PIXELS_PER_IMAGE: usize = 28 * 28;
const NEURAL_NETWORK_JSON_FILENAME: &str = "network.json";

fn main() -> std::result::Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    println!("Arguments: {:?}", args);

    if args.len() < 2 {
        return make_error_str("An argument wasn't passed in. Run with 'cargo run -- <argument>");
    }

    match args[1].as_str() {
        "train" => train(),
        "test" => test(),
        _ => return make_error(std::format!("Didn't recognize argument: {0}", args[1])),
    }

    Ok(())
}

fn make_error_str(message: &'static str) -> std::result::Result<(), String> {
    return std::result::Result::Err(std::string::String::from(message));
}

fn make_error(message: std::string::String) -> std::result::Result<(), String> {
    return std::result::Result::Err(message);
}

fn train() {
    std::println!("Load training data");
    let examples = load_data(
        "data/train-labels.idx1-ubyte",
        "data/train-images.idx3-ubyte",
    );

    let mut neural_network = nn::NN::new(
        &[PIXELS_PER_IMAGE as u32, 100, 10],
        nn::Activation::Sigmoid,
        nn::Activation::Sigmoid,
    );

    std::println!("Begin training");

    let now = std::time::Instant::now();

    neural_network
        .train(&examples)
        .halt_condition(nn::HaltCondition::Epochs(3))
        .log_interval(Some(1))
        .rate(0.3)
        .go();

    let elapsed_time = now.elapsed();
    std::println!(
        "Training finished. took {} seconds.",
        elapsed_time.as_secs()
    );

    let nn_json = neural_network.to_json();

    std::fs::write(NEURAL_NETWORK_JSON_FILENAME, nn_json).unwrap();
}

fn test() {
    std::println!("Load test data");
    let examples = load_data("data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte");

    std::println!("Load neural network json");
    let mut f = std::fs::File::open(NEURAL_NETWORK_JSON_FILENAME).unwrap();

    let mut nn_json = std::string::String::new();
    f.read_to_string(&mut nn_json).unwrap();

    std::println!("Load neural network from json");
    let neural_network = nn::NN::from_json(nn_json.as_str());

    std::println!("Begin test");

    let mut fails = 0;

    for test_pair in &examples {
        let result = neural_network.run(test_pair.0.as_slice());

        if evaluate_label_vector(&result) != evaluate_label_vector(&test_pair.1) {
            fails += 1;
        }
    }

    std::println!(
        "Finished test. {0} cases. {1} failures.",
        examples.len(),
        fails
    );
}

fn evaluate_label_vector(label_vec: &std::vec::Vec<f64>) -> u8 {
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
    return index_of_max_value as u8;
}

fn load_data(
    labels_file_path: &str,
    images_file_path: &str,
) -> std::vec::Vec<(std::vec::Vec<f64>, std::vec::Vec<f64>)> {
    std::println!("Open files");
    let labels = open_labels(labels_file_path);

    let image_bytes = open_images(images_file_path);

    std::println!("Transform data");
    let mut examples: std::vec::Vec<(std::vec::Vec<f64>, std::vec::Vec<f64>)> =
        std::vec::Vec::with_capacity(labels.len());

    for (item_index, label) in labels.iter().enumerate() {
        let image_start_index = item_index * PIXELS_PER_IMAGE;
        let image_end_index = (item_index + 1) * PIXELS_PER_IMAGE;

        let mut image_vec = std::vec::Vec::with_capacity(PIXELS_PER_IMAGE);
        for pixel in &image_bytes[image_start_index..image_end_index] {
            image_vec.push((*pixel as f64) / 255f64);
        }
        let output_vector = make_output_vector_from_label(*label);
        examples.push((image_vec, output_vector));
    }

    return examples;
}

fn make_output_vector_from_label(label: u8) -> std::vec::Vec<f64> {
    let mut vector = vec![0f64; 10];
    vector[label as usize] = 1f64;
    return vector;
}

fn open_labels(path: &str) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap();

    let magic_number = read_u32(&mut f);
    assert_eq!(magic_number, 2049);

    let number_of_items = read_u32(&mut f);
    assert!(number_of_items == 10000 || number_of_items == 60000);

    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).unwrap();

    return bytes;
}

fn open_images(path: &str) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap();

    let magic_number = read_u32(&mut f);
    assert_eq!(magic_number, 2051);

    let number_of_items = read_u32(&mut f);
    assert!(number_of_items == 10000 || number_of_items == 60000);

    let number_of_rows = read_u32(&mut f);
    assert_eq!(number_of_rows, 28);

    let number_of_columns = read_u32(&mut f);
    assert_eq!(number_of_columns, 28);

    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).unwrap();

    return bytes;
}

fn read_u32(file: &mut std::fs::File) -> u32 {
    let mut buffer = [0; 4];
    let num_bytes_read = file.read(&mut buffer).unwrap();
    assert!(num_bytes_read == 4);
    return u32::from_be_bytes(buffer);
}

fn print_image(buffer: &Vec<f64>, number_of_rows: u32, number_of_columns: u32) {
    std::println!("Image:");
    for row in 0..number_of_rows {
        for column in 0..number_of_columns {
            let index = (row * number_of_columns + column) as usize;
            std::print!("{:3} ", buffer[index]);
        }
        std::print!("\n");
    }
}

#[wasm_bindgen]
pub fn recognize(neural_network_json_text: &str, image_bytes: &[u8]) -> u8 {
    std::println!("Load neural network from json");
    let neural_network = nn::NN::from_json(neural_network_json_text);

    let mut input_vec = std::vec::Vec::with_capacity(PIXELS_PER_IMAGE);
    for pixel_index in 0..PIXELS_PER_IMAGE {
        input_vec.push((image_bytes[pixel_index] as f64) / 255f64);
    }

    let result_vector = neural_network.run(input_vec.as_slice());

    return evaluate_label_vector(&result_vector);
}
