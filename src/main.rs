use std::io::prelude::*;

fn main() {
    open_labels("data/train-labels.idx1-ubyte");
    open_labels("data/t10k-labels.idx1-ubyte");
}

fn open_labels(path: &str) {
    let mut f = std::fs::File::open(path).unwrap();

    let mut magic_number_buffer = [0; 4];
    let num_bytes_read = f.read(&mut magic_number_buffer).unwrap();
    assert!(num_bytes_read == 4);
    let labels_magic_number = u32::from_be_bytes(magic_number_buffer);
    assert!(labels_magic_number == 2049);

    let mut number_of_items_buffer = [0; 4];
    let num_bytes_read = f.read(&mut number_of_items_buffer).unwrap();
    assert!(num_bytes_read == 4);
    let number_of_items = u32::from_be_bytes(number_of_items_buffer);
    assert!(number_of_items == 10000 || number_of_items == 60000);

    for _ in 0..10 {
        let mut label_buffer = [0];
        let num_bytes_read = f.read(&mut label_buffer).unwrap();
        assert!(num_bytes_read == 1);
        let label = label_buffer[0];
        assert!(label <= 9);
        std::println!("{}", label)
    }
}
