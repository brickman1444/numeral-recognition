use std::io::prelude::*;

fn main() {
    let mut f = std::fs::File::open("data/t10k-labels.idx1-ubyte").unwrap();
    let mut buffer = [0; 8];

    let num_bytes_read = f.read(&mut buffer).unwrap();
    assert!(num_bytes_read == 8)
}
