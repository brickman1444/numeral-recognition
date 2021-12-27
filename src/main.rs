use std::io::prelude::*;

fn main() {
    let mut f = std::fs::File::open("data/t10k-labels.idx1-ubyte").unwrap();

    let mut magic_number_buffer = [0; 4];
    let num_bytes_read = f.read(&mut magic_number_buffer).unwrap();
    assert!(num_bytes_read == 4);
    let labels_magic_number = u32::from_be_bytes(magic_number_buffer);
    assert!(labels_magic_number == 2049);

    let mut number_of_items_buffer = [0; 4];
    let num_bytes_read = f.read(&mut number_of_items_buffer).unwrap();
    assert!(num_bytes_read == 4);
    let number_of_items = u32::from_be_bytes(number_of_items_buffer);
    assert!(number_of_items == 10000);
}
