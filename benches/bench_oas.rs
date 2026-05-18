#![feature(generic_const_exprs)]

// rust-side benches are a bit more precise and bypass python <> rust overhead

use criterion::black_box;
use giga_levenshtein::simd::bitty_levenshtein_simd_by_n_limited;

// currently: 2048 x 2048, dist=8 -> 6 seconds
fn main() {
    const CHUNK_SIZE: usize = 256;
    let max_dist = 8;

    // read heavy_seqs.txt.gz, which has antibody on each line, save to variable
    let file =
        std::fs::File::open("benches/heavy_seqs.txt.gz").expect("Could not open heavy_seqs.txt.gz");
    let gz = flate2::read::GzDecoder::new(file);
    let reader = std::io::BufReader::new(gz);
    let heavy_seq_owned: Vec<String> = std::io::BufRead::lines(reader)
        .map(|l| l.expect("Could not read line").trim().to_string())
        .collect::<Vec<String>>()
        .as_slice()[0..2048]
        .to_vec();

    let queries: Vec<&[u8]> = heavy_seq_owned.iter().map(|s| s.as_bytes()).collect();
    let targets: Vec<&[u8]> = heavy_seq_owned.iter().map(|s| s.as_bytes()).collect();

    let mut all_results: Vec<Vec<i32>> = vec![vec![0; CHUNK_SIZE]; heavy_seq_owned.len()];

    let start_time = std::time::Instant::now();
    let _ = {
        for q_chunk in queries.chunks(CHUNK_SIZE) {
            if q_chunk.len() == CHUNK_SIZE {
                let q_arr: &[&[u8]; CHUNK_SIZE] = q_chunk.try_into().unwrap();
                let results =
                    bitty_levenshtein_simd_by_n_limited::<CHUNK_SIZE>(q_arr, &targets, max_dist);
                all_results.extend_from_slice(&results);
            } // skip the last chunk
        }

        black_box(all_results);
    };
    let elapsed = start_time.elapsed();

    println!("Elapsed: {} ms", elapsed.as_millis());
}
