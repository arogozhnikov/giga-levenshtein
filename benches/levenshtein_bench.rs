#![feature(portable_simd)]
#![feature(generic_const_exprs)]

// rust-side benches are a bit more precise and bypass python <> rust overhead

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use giga_levenshtein::simd::{
    bitty_levenshtein_n_by_1, bitty_levenshtein_simd_by_1_limited,
    bitty_levenshtein_simd_by_n_limited,
};

const N_SEQUENCE_OPTIONS: &[usize] = &[256usize];
const SEQ_LEN_OPTIONS: &[usize] = &[256usize];

fn random_byte_seq(rng: &mut StdRng, len: usize) -> Vec<u8> {
    (0..len).map(|_| rng.gen_range(b'a'..=b'z')).collect()
}

fn random_byte_seqs(rng: &mut StdRng, count: usize, len: usize) -> Vec<Vec<u8>> {
    (0..count).map(|_| random_byte_seq(rng, len)).collect()
}

fn random_byte_seqs_range(
    rng: &mut StdRng,
    count: usize,
    min_len: usize,
    max_len: usize,
) -> Vec<Vec<u8>> {
    (0..count)
        .map(|_| random_byte_seq(&mut rng.clone(), rng.gen_range(min_len..=max_len)))
        .collect()
}

fn bench_scalar_1_to_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_1_to_n");
    let seed: u64 = 42;

    for &n_seqs in N_SEQUENCE_OPTIONS {
        for &seq_len in SEQ_LEN_OPTIONS {
            let mut rng = StdRng::seed_from_u64(seed);
            let query = random_byte_seq(&mut rng, seq_len);
            let targets = random_byte_seqs(&mut rng, n_seqs, seq_len);

            let param = format!("seqs={n_seqs}_len={seq_len}");
            group.bench_function(BenchmarkId::new("scalar", &param), |b| {
                b.iter(|| {
                    let results: Vec<usize> = targets
                        .iter()
                        .map(|t| levenshtein_naive(black_box(&query), black_box(t.as_slice())))
                        .collect();
                    black_box(results);
                })
            });
        }
    }
    group.finish();
}

fn levenshtein_naive(a: &[u8], b: &[u8]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Ensure we iterate over the shorter dimension for memory efficiency.
    if m < n {
        return levenshtein_naive(b, a);
    }

    // `prev` holds the previous row of the DP matrix.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

fn bench_bitty_simd_1_to_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitty_simd_1_to_n");
    let seed: u64 = 42;

    for &n_seqs in N_SEQUENCE_OPTIONS {
        for &seq_len in SEQ_LEN_OPTIONS {
            let mut rng = StdRng::seed_from_u64(seed);
            let query = random_byte_seq(&mut rng, seq_len);
            let targets = random_byte_seqs(&mut rng, n_seqs, seq_len);

            let param = format!("seqs={n_seqs}_len={seq_len}");
            group.bench_function(BenchmarkId::new("bitty_simd", &param), |b| {
                b.iter(|| {
                    let slices: Vec<&[u8]> = targets.iter().map(|t| t.as_slice()).collect();
                    let results = bitty_levenshtein_n_by_1(black_box(&slices), black_box(&query));
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

                let query = random_byte_seq(&mut rng, seq_len);
                let targets = random_byte_seqs(&mut rng, n_seqs, seq_len);

                let param = format!("seqs={n_seqs}_len={seq_len}_maxd={max_dist}");

                group.bench_function(BenchmarkId::new("bitty_limited", &param), |b| {
                    b.iter(|| {
                        let slices: Vec<&[u8]> = targets.iter().map(|t| t.as_slice()).collect();
                        let mut all_results: Vec<i32> = Vec::with_capacity(n_seqs);
                        for chunk in slices.chunks(128) {
                            if chunk.len() == 128 {
                                let arr: &[&[u8]; 128] = chunk.try_into().unwrap();
                                let results = bitty_levenshtein_simd_by_1_limited::<128>(
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

fn bench_bitty_simd_limited_m_to_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitty_simd_limited_m_to_n");
    let seed: u64 = 42;

    for &n_seqs in N_SEQUENCE_OPTIONS {
        for &seq_len in SEQ_LEN_OPTIONS {
            for &max_dist in &[8usize, 32, 128] {
                let mut rng = StdRng::seed_from_u64(seed);
                let queries_tmp = random_byte_seqs_range(&mut rng, n_seqs, seq_len / 2, seq_len);
                let queries: Vec<&[u8]> = queries_tmp.iter().map(|t| t.as_slice()).collect();
                let targets_tmp = random_byte_seqs_range(&mut rng, 128, seq_len / 2, seq_len);
                let targets: Vec<&[u8]> = targets_tmp.iter().map(|x| x.as_slice()).collect();

                let param = format!("seqs={n_seqs}_len={seq_len}_maxd={max_dist}");

                const CHUNK_SIZE: usize = 256;
                group.bench_function(BenchmarkId::new("bitty_limited", &param), |b| {
                    b.iter(|| {
                        let mut all_results: Vec<Vec<i32>> = vec![vec![0; CHUNK_SIZE]; n_seqs];
                        for q_chunk in queries.chunks(CHUNK_SIZE) {
                            if q_chunk.len() == CHUNK_SIZE {
                                let q_arr: &[&[u8]; CHUNK_SIZE] = q_chunk.try_into().unwrap();
                                let results = bitty_levenshtein_simd_by_n_limited::<CHUNK_SIZE>(
                                    black_box(q_arr),
                                    black_box(&targets),
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
    bench_bitty_simd_1_to_n,
    bench_bitty_simd_limited_1_to_n,
    bench_bitty_simd_limited_m_to_n,
);
criterion_main!(benches);
