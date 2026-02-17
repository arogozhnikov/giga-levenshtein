use std::array;
use std::simd::Mask;
// use std::array::from_fn;
// use std::io::empty;
use std::simd::cmp::SimdOrd;
use std::simd::cmp::SimdPartialEq;
use std::simd::{Select, Simd};
use std::time::Instant;

fn levenshtein(a: &str, b: &str) -> usize {
    let n = b.len();
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];
    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

type U8<const N: usize> = Simd<u8, N>;

#[allow(non_snake_case)]
fn levenshtein_simd_by_1<const N: usize>(a: &[&[u8]; N], b: &[u8]) -> [u8; N] {
    // assumes all lengths are identical, not checked right now
    let (I, J) = (a[0].len(), b.len());
    let mut prev: Vec<U8<N>> = (0..=J).map(|i| U8::<N>::splat(i as u8)).collect();
    let mut curr = vec![U8::<N>::splat(0); J + 1];
    let one = U8::<N>::splat(1);

    for i in 1..=I {
        curr[0] = U8::<N>::splat(i as u8);

        let c_a = U8::<N>::from_array(std::array::from_fn(|s| a[s][i - 1] as u8));

        for j in 1..=J {
            let mask = c_a.simd_eq(U8::<N>::splat(b[j - 1]));

            curr[j] = (mask.select(prev[j - 1], prev[j - 1] + one))
                .simd_min(prev[j] + one)
                .simd_min(curr[j - 1] + one)
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    return *prev[J].as_array();
}

fn select<const N: usize>(cond: U8<N>, first: U8<N>, second: U8<N>) -> U8<N> {
    return (first & cond) | (second & !cond);
}

pub fn levenshtein_n_by_1(a: Vec<&[u8]>, b: &[u8]) -> Vec<u8> {
    assert!(!b.contains(&255));
    a.chunks(32)
        .flat_map(|chunk| {
            if chunk.len() == 32 {
                levenshtein_simd_by_1::<32>(chunk.try_into().unwrap(), b).to_vec()
            } else {
                panic!("chunk len is not 32");
            }
        })
        .collect()
}

#[allow(non_snake_case)]
fn levenshtein_n_by_8<const N: usize>(a: [&[u8]; N], b: [&[u8]; 8]) -> [u8; N] {
    let (I, J) = (a[0].len(), b.len());
    // TODO init is totally bad, dx=+1, dy=?
    let mut prev: Vec<U8<N>> = (0..4 * (J + 1)).map(|i| U8::<N>::splat(i as u8)).collect();
    let mut curr = vec![U8::<N>::splat(0); 4 * (J + 1)];

    let mut bit_shifts: Vec<U8<N>> = vec![]; // 1 for each mismatch

    let one = U8::<N>::splat(1);
    let zeroes = U8::<N>::splat(0);

    for i in 1..=I {
        let mut is_same = zeroes;
        let c_a = U8::<N>::from_array(std::array::from_fn(|s| a[s][i - 1] as u8));

        let mut dy_p = one;
        let mut dy_n = zeroes;

        for j in 1..=J {
            for shift in 0..8 {
                is_same = is_same << 1;
                is_same = c_a
                    .simd_eq(U8::<N>::splat(b[shift][j - 1]))
                    .select(is_same + one, is_same);
            }
            let (dxp, dxn) = (prev[2 * j + 0], prev[2 * j + 1]);

            let diag0 = is_same | dxn | dy_n;

            curr[4 * j + 0] = select(diag0, dy_n, !dy_n); // dxp
            curr[4 * j + 1] = select(diag0, dy_p, zeroes); // dxn
            dy_p = select(diag0, dxn, !dxn);
            dy_n = select(diag0, dxp, zeroes);
            if j == i {
                bit_shifts.push(diag0);
            }
            if j >= I {
                bit_shifts.push(dy_p);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    // TODO put somewhere the last condition.

    return *prev[J].as_array();
}

fn compute_sum<const N: usize>(a: &[U8<N>]) -> [u16; N] {
    let mut counters: [Simd<u8, N>; 8] = array::from_fn(|_| U8::<N>::splat(0));
    let mask = Simd::<u8, N>::splat(1);
    let mut result: [u16; N] = std::array::from_fn(|_i| 0u16);

    for i in 0..a.len() {
        let item = a[i];
        for i in 0..8 {
            counters[i] += (item >> (i as u8)) & mask;
        }
        if (i % 128 == 0) || (i + 1 == a.len()) {
            for i in 0..N {
                result[i * 8..(i + 1) * N]
                    .iter_mut()
                    .zip(&counters[i].to_array())
                    .for_each(|(x, y)| *x += *y as u16);
            }
            counters = array::from_fn(|_| U8::<N>::splat(0)); // reset
        }
    }
    // dump rest of counters to result

    return result;
}

fn main() {
    let a = "abcd1234".repeat(1024 / 8);
    let mut b = a.clone();
    b.replace_range(512..514, "ee");

    for _i in 0..3 {
        let start = Instant::now();
        let dist = levenshtein(&a, &b);
        let elapsed = start.elapsed();
        println!("Time elapsed 1: {:?} {:?}", elapsed, dist);

        let lambda = |i| {
            if i % 2 == 0 {
                a.as_bytes()
            } else {
                b.as_bytes()
            }
        };

        let a_seqs: [&[u8]; 8] = array::from_fn(lambda);
        let start = Instant::now();
        let dist = levenshtein_n_by_1(a_seqs.to_vec(), b.as_bytes());
        let elapsed = start.elapsed();

        println!("Time elapsed 2: {:?} {:?}", elapsed, dist[0]);

        let a_seqs: [&[u8]; 16] = array::from_fn(lambda);
        let start = Instant::now();
        let dist = levenshtein_n_by_1(a_seqs.to_vec(), b.as_bytes());
        let elapsed = start.elapsed();

        println!("Time elapsed 3: {:?} {:?}", elapsed, dist[0]);

        let a_seqs: [&[u8]; 32] = array::from_fn(lambda);
        let start = Instant::now();
        let dist = levenshtein_n_by_1(a_seqs.to_vec(), b.as_bytes());
        let elapsed = start.elapsed();

        // println!("Levenshtein distance: {}", dist);
        println!("Time elapsed 4: {:?} {:?}", elapsed, dist[0]);

        let a_seqs: [&[u8]; 32] = array::from_fn(lambda);
        let b_seqs: [&[u8]; 8] = array::from_fn(lambda);
        let start = Instant::now();
        let dist = levenshtein_n_by_8(a_seqs, b_seqs);
        let elapsed = start.elapsed();
        println!("Time elapsed 5: {:?} {:?}", elapsed, dist[0]);
    }
}
