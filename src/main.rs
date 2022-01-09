use numeral_recognition::*;
use std::io::prelude::*;

const NEURAL_NETWORK_JSON_FILENAME: &str = "network.json";

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    println!("Arguments: {:?}", args);

    if args.len() < 2 {
        return Err("An argument wasn't passed in. Run with 'cargo run -- <argument>".into());
    }

    match args[1].as_str() {
        "train" => train(),
        "test" => test(),
        _ => return Err(std::format!("Didn't recognize argument: {0}", args[1])),
    }

    Ok(())
}

fn train() {
    println!("Load training data");
    let examples = load_data(
        "data/train-labels.idx1-ubyte",
        "data/train-images.idx3-ubyte",
    );

    let mut neural_network = nn::NN::new(
        &[PIXELS_PER_IMAGE as u32, 100, 10],
        nn::Activation::Sigmoid,
        nn::Activation::Sigmoid,
    );

    println!("Begin training");

    let now = std::time::Instant::now();

    neural_network
        .train(&examples)
        .halt_condition(nn::HaltCondition::Epochs(3))
        .log_interval(Some(1))
        .rate(0.3)
        .go();

    let elapsed_time = now.elapsed();
    println!(
        "Training finished. took {} seconds.",
        elapsed_time.as_secs()
    );

    let nn_json = neural_network.to_json();

    std::fs::write(NEURAL_NETWORK_JSON_FILENAME, nn_json).unwrap();
}

fn test() {
    println!("Load test data");
    let examples = load_data("data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte");

    println!("Load neural network json");
    let mut f = std::fs::File::open(NEURAL_NETWORK_JSON_FILENAME).unwrap();

    let mut nn_json = String::new();
    f.read_to_string(&mut nn_json).unwrap();

    println!("Load neural network from json");
    let neural_network = nn::NN::from_json(nn_json.as_str());

    println!("Begin test");

    let mut fails = 0;

    for test_pair in &examples {
        let result = neural_network.run(test_pair.0.as_slice());

        if evaluate_label_vector(&result) != evaluate_label_vector(&test_pair.1) {
            fails += 1;
        }
    }

    println!(
        "Finished test. {0} cases. {1} failures.",
        examples.len(),
        fails
    );
}

fn load_data(labels_file_path: &str, images_file_path: &str) -> Vec<(Vec<f64>, Vec<f64>)> {
    println!("Open files");
    let labels = open_labels(labels_file_path);

    let image_bytes = open_images(images_file_path);

    println!("Transform data");

    labels
        .iter()
        .enumerate()
        .map(|(item_index, label)| {
            let image_start_index = item_index * PIXELS_PER_IMAGE;
            let image_end_index = (item_index + 1) * PIXELS_PER_IMAGE;

            let image_vec = image_bytes[image_start_index..image_end_index]
                .iter()
                .map(|pixel| (*pixel as f64) / 255f64)
                .collect();

            let output_vector = make_output_vector_from_label(*label);
            (image_vec, output_vector)
        })
        .collect()
}

fn make_output_vector_from_label(label: u8) -> Vec<f64> {
    let mut vector = vec![0f64; 10];
    vector[label as usize] = 1f64;
    vector
}

fn open_labels(path: &str) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap();

    let magic_number = read_u32(&mut f);
    assert_eq!(magic_number, 2049);

    let number_of_items = read_u32(&mut f);
    assert!(number_of_items == 10000 || number_of_items == 60000);

    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).unwrap();

    bytes
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

    bytes
}

fn read_u32(file: &mut std::fs::File) -> u32 {
    let mut buffer = [0; 4];
    let num_bytes_read = file.read(&mut buffer).unwrap();
    assert!(num_bytes_read == 4);
    u32::from_be_bytes(buffer)
}
