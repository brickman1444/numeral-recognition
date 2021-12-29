use std::io::prelude::*;

fn main() {
    std::println!("Open files");
    let labels = open_labels("data/train-labels.idx1-ubyte");
    //open_labels("data/t10k-labels.idx1-ubyte");

    let image_bytes = open_images("data/train-images.idx3-ubyte");
    //open_images("data/t10k-images.idx3-ubyte");

    let pixels_per_image = 28 * 28;

    std::println!("Construct training data");
    let mut examples : std::vec::Vec<(std::vec::Vec<f64>, std::vec::Vec<f64>)> = std::vec::Vec::with_capacity(labels.len());

    for (item_index, label) in labels.iter().enumerate() {
        let image_start_index = item_index * pixels_per_image;
        let image_end_index = (item_index + 1) * pixels_per_image;

        let mut image_vec = std::vec::Vec::with_capacity(pixels_per_image);
        for pixel in &image_bytes[image_start_index..image_end_index] {
            image_vec.push(*pixel as f64);
        }
        let output_vector = make_output_vector_from_label(*label);
        examples.push(( image_vec, output_vector));
    }

    let mut neural_network = nn::NN::new(&[pixels_per_image as u32, 10]); // TODO: Optimize neural network parameters here

    std::println!("Begin training");
    neural_network.train(&examples)
        .halt_condition( nn::HaltCondition::Epochs(1) )
        .log_interval( Some(1) )
        .rate( 0.3 )
        .go();
    std::println!("Training finished");
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

    return bytes
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
