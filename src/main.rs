use std::io::prelude::*;

fn main() {
    open_labels("data/train-labels.idx1-ubyte");
    open_labels("data/t10k-labels.idx1-ubyte");

    open_images("data/train-images.idx3-ubyte");
    open_images("data/t10k-images.idx3-ubyte");
}

fn open_labels(path: &str) {
    let mut f = std::fs::File::open(path).unwrap();

    let magic_number = read_u32(&mut f);
    assert!(magic_number == 2049);

    let number_of_items = read_u32(&mut f);
    assert!(number_of_items == 10000 || number_of_items == 60000);

    std::println!("Labels:");
    for _ in 0..10 {
        let label = read_byte(&mut f);
        assert!(label <= 9);
        std::println!("{}", label)
    }
}

fn open_images(path: &str) {
    let mut f = std::fs::File::open(path).unwrap();

    let magic_number = read_u32(&mut f);
    assert!(magic_number == 2051);

    let number_of_items = read_u32(&mut f);
    assert!(number_of_items == 10000 || number_of_items == 60000);

    let number_of_rows = read_u32(&mut f);
    assert!(number_of_rows == 28);

    let number_of_columns = read_u32(&mut f);
    assert!(number_of_columns == 28);

    let mut images = Vec::new();
    f.read_to_end(&mut images).unwrap();

    std::println!("Image:");
    for row in 0..number_of_rows {
        for column in 0..number_of_columns {
            let index = (row * number_of_columns + column) as usize;
            std::print!("{:3} ", images[index]);
        }
        std::print!("\n");
    }
}

fn read_u32(file: &mut std::fs::File) -> u32 {
    let mut buffer = [0; 4];
    let num_bytes_read = file.read(&mut buffer).unwrap();
    assert!(num_bytes_read == 4);
    return u32::from_be_bytes(buffer);
}

fn read_byte(file: &mut std::fs::File) -> u8 {
    let mut buffer = [0];
    let num_bytes_read = file.read(&mut buffer).unwrap();
    assert!(num_bytes_read == 1);
    return buffer[0];
}
