#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rust_levenshtein::levenshtein_bytes;
use rust_levenshtein::simd::{
    bitty_levenshtein_n_by_1, bitty_levenshtein_simd_by_1_limited, levenshtein_n_by_1,
};

const N_SEQUENCE_OPTIONS: &[usize] = &[256usize];
const SEQ_LEN_OPTIONS: &[usize] = &[256usize];

/// Generate a random byte sequence of given length using lowercase ASCII letters.
fn random_seq(rng: &mut StdRng, len: usize) -> Vec<u8> {
    (0..len).map(|_| rng.gen_range(b'a'..=b'z')).collect()
}

/// Generate `count` random sequences, all of the same `len`.
fn random_seqs(rng: &mut StdRng, count: usize, len: usize) -> Vec<Vec<u8>> {
    (0..count).map(|_| random_seq(rng, len)).collect()
}

fn bench_scalar_1_to_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_1_to_n");
    let seed: u64 = 42;

    for &n_seqs in N_SEQUENCE_OPTIONS {
        for &seq_len in SEQ_LEN_OPTIONS {
            let mut rng = StdRng::seed_from_u64(seed);
            let query = random_seq(&mut rng, seq_len);
            let targets = random_seqs(&mut rng, n_seqs, seq_len);

            let param = format!("seqs={n_seqs}_len={seq_len}");
            group.bench_function(BenchmarkId::new("scalar", &param), |b| {
                b.iter(|| {
                    let results: Vec<usize> = targets
                        .iter()
                        .map(|t| levenshtein_bytes(black_box(&query), black_box(t.as_slice())))
                        .collect();
                    black_box(results);
                })
            });
        }
    }
    group.finish();
}

fn bench_simd_1_to_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_dp_1_to_n");
    let seed: u64 = 42;

    for &n_seqs in N_SEQUENCE_OPTIONS {
        for &seq_len in SEQ_LEN_OPTIONS {
            let mut rng = StdRng::seed_from_u64(seed);
            let query = random_seq(&mut rng, seq_len);
            let targets = random_seqs(&mut rng, n_seqs, seq_len);

            let param = format!("seqs={n_seqs}_len={seq_len}");
            group.bench_function(BenchmarkId::new("simd_dp", &param), |b| {
                b.iter(|| {
                    let slices: Vec<&[u8]> = targets.iter().map(|t| t.as_slice()).collect();
                    let results = levenshtein_n_by_1(black_box(slices), black_box(&query));
                    black_box(results);
                })
            });
        }
    }
    group.finish();
}

fn bench_bitty_simd_1_to_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitty_simd_1_to_n");
    let seed: u64 = 42;

    for &n_seqs in N_SEQUENCE_OPTIONS {
        for &seq_len in SEQ_LEN_OPTIONS {
            let mut rng = StdRng::seed_from_u64(seed);
            let query = random_seq(&mut rng, seq_len);
            let targets = random_seqs(&mut rng, n_seqs, seq_len);

            let param = format!("seqs={n_seqs}_len={seq_len}");
            group.bench_function(BenchmarkId::new("bitty_simd", &param), |b| {
                b.iter(|| {
                    let slices: Vec<&[u8]> = targets.iter().map(|t| t.as_slice()).collect();
                    let results = bitty_levenshtein_n_by_1(black_box(slices), black_box(&query));
                    black_box(results);
                })
            });
        }
    }
    group.finish();
}

fn bench_bitty_simd_limited_1_to_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitty_simd_limited_1_to_n");
    let seed: u64 = 42;

    for &n_seqs in N_SEQUENCE_OPTIONS {
        for &seq_len in SEQ_LEN_OPTIONS {
            for &max_dist in &[8usize, 32, 128] {
                let mut rng = StdRng::seed_from_u64(seed);
                let query = random_seq(&mut rng, seq_len);
                let targets = random_seqs(&mut rng, n_seqs, seq_len);

                let param = format!("seqs={n_seqs}_len={seq_len}_maxd={max_dist}");

                // We call bitty_levenshtein_simd_by_1_limited in chunks of 128
                // (const N=16, M=128) matching the main code's usage.
                group.bench_function(BenchmarkId::new("bitty_limited", &param), |b| {
                    b.iter(|| {
                        let slices: Vec<&[u8]> = targets.iter().map(|t| t.as_slice()).collect();
                        let mut all_results: Vec<u8> = Vec::with_capacity(n_seqs);
                        for chunk in slices.chunks(128) {
                            if chunk.len() == 128 {
                                let arr: &[&[u8]; 128] = chunk.try_into().unwrap();
                                let results = bitty_levenshtein_simd_by_1_limited::<16, 128>(
                                    black_box(arr),
                                    black_box(&query),
                                    black_box(max_dist),
                                );
                                all_results.extend_from_slice(&results);
                            }
                        }
                        black_box(all_results);
                    })
                });
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_1_to_n,
    bench_simd_1_to_n,
    bench_bitty_simd_1_to_n,
    bench_bitty_simd_limited_1_to_n,
);
criterion_main!(benches);
