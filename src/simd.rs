use std::simd::cmp::SimdOrd;
use std::simd::cmp::SimdPartialEq;
use std::simd::{Select, Simd};

fn _levenshtein(a: &str, b: &str) -> usize {
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

fn _bitty_levenshtein(a: &[u8], b: &[u8]) -> u8 {
    // myers-style algo
    if b.len() == 0 {
        return a.len() as u8;
    };
    if a.len() == 0 {
        return b.len() as u8;
    }
    let n = b.len();
    let mut prev_hp = vec![true; n]; // all 1
    let mut prev_hn = vec![false; n];
    //
    let mut curr_hp = vec![true; n];
    let mut curr_hn = vec![false; n];

    for (i, ca) in a.iter().enumerate() {
        let mut curr_dp_j = false;

        for (j, cb) in b.iter().enumerate() {
            let is_match = ca == cb;
            let curr_vp_j: bool;
            let curr_vn_j: bool;
            if j == 0 {
                curr_vp_j = true;
                curr_vn_j = false;
            } else {
                // curr_d[j - i] = prev_h[ j - 1 ] + curr_v [ j ]
                // res := curr_dp[j - 1] - prev_hp[j - 1] + prev_hn[j - 1];
                let curr_dp_j_m1 = curr_dp_j;
                curr_vp_j = prev_hn[j - 1] | (curr_dp_j_m1 & !prev_hp[j - 1]); // res > 0
                curr_vn_j = prev_hp[j - 1] & !curr_dp_j_m1; // res < 0
            }
            // curr_d[j], before we used previous variable
            curr_dp_j = !(prev_hn[j] | curr_vn_j | is_match);
            {
                // curr_h[j] = curr_d[j] - curr_v[j]
                // res := curr_dp[j] - curr_vp[j] + curr_vn[j];
                curr_hp[j] = (curr_dp_j & !curr_vp_j) | curr_vn_j; // res > 0
                curr_hn[j] = curr_vp_j & !curr_dp_j; // res < 0
            }
        }
        if i + 1 < a.len() {
            std::mem::swap(&mut prev_hn, &mut curr_hn);
            std::mem::swap(&mut prev_hp, &mut curr_hp);
        }
    }

    let result = (a.len() as i32) + curr_hp.iter().map(|x| *x as i32).sum::<i32>()
        - curr_hn.iter().map(|x| *x as i32).sum::<i32>();
    result as u8
}

type U8<const N: usize> = Simd<u8, N>;

pub fn bitty_levenshtein_simd_by_1<const N: usize, const M: usize>(
    a: &[&[u8]; M],
    b: &[u8],
) -> [u8; M] {
    // assumes all lengths are identical, does not check this right now
    let (alen, blen) = (a[0].len(), b.len());

    if blen == 0 {
        return [alen as u8; M];
    };
    if alen == 0 {
        return [blen as u8; M];
    }
    assert!(N * 8 == M);

    // myers-style algo
    let mut prev_hp = vec![U8::<N>::splat(255); blen];
    let mut prev_hn = vec![U8::<N>::splat(0); blen];
    //
    let mut curr_hp = vec![U8::<N>::splat(255); blen];
    let mut curr_hn = vec![U8::<N>::splat(0); blen];

    for i in 0..alen {
        let mut curr_dnp_j = U8::<N>::splat(255);

        let c_a: [U8<N>; 8] = std::array::from_fn(|shift| {
            U8::<N>::from_array(std::array::from_fn(|s| a[shift + 8 * s][i] as u8))
        });

        for (j, cb) in b.iter().enumerate() {
            let mut is_match = U8::<N>::splat(0);
            for shift in 0..8 {
                is_match |= c_a[shift]
                    .simd_eq(U8::<N>::splat(*cb))
                    .select(U8::<N>::splat(1 << shift), U8::<N>::splat(0));
            }
            let curr_vp_j: U8<N>;
            let curr_vn_j: U8<N>;
            if j == 0 {
                curr_vp_j = U8::<N>::splat(255);
                curr_vn_j = U8::<N>::splat(0);
            } else {
                // curr_d[j - i] = prev_h[ j - 1 ] + curr_v [ j ]
                // res := curr_dp[j - 1] - prev_hp[j - 1] + prev_hn[j - 1];
                let curr_dnp_j_m1 = curr_dnp_j;
                curr_vp_j = prev_hn[j - 1] | !(curr_dnp_j_m1 | prev_hp[j - 1]); // res > 0
                curr_vn_j = prev_hp[j - 1] & curr_dnp_j_m1; // res < 0
            }
            // curr_d[j], before we used previous variable
            curr_dnp_j = prev_hn[j] | curr_vn_j | is_match;
            {
                // curr_h[j] = curr_d[j] - curr_v[j]
                // res := curr_dp[j] - curr_vp[j] + curr_vn[j];
                curr_hp[j] = !(curr_dnp_j | curr_vp_j) | curr_vn_j; // res > 0
                curr_hn[j] = curr_vp_j & curr_dnp_j; // res < 0
            }
        }
        if i + 1 < alen {
            std::mem::swap(&mut prev_hn, &mut curr_hn);
            std::mem::swap(&mut prev_hp, &mut curr_hp);
        }
    }

    let one = U8::<N>::splat(1);
    let mut result = [0u8; M];
    for shift in 0..8 {
        let result_for_shift: U8<N> = U8::<N>::splat(alen as u8)
            + curr_hp.iter().map(|x| (*x >> shift) & one).sum::<U8<N>>()
            - curr_hn.iter().map(|x| (*x >> shift) & one).sum::<U8<N>>();

        for s in 0..N {
            result[shift as usize + s * 8] = result_for_shift[s];
        }
    }

    result
}

pub fn bitty_levenshtein_simd_by_1_limited<const N: usize, const M: usize>(
    a: &[&[u8]; M],
    b: &[u8],
    max_dist: usize,
) -> [u8; M] {
    // assumes all lengths are identical, does not check this right now
    // remark: pre-computing is_match is slower if max_dist is smaller than dictionary
    let (alen, blen) = (a[0].len(), b.len());
    assert!(max_dist <= 254);
    if (alen + max_dist < blen) || (blen + max_dist < alen) {
        return [(max_dist + 1) as u8; M];
    }

    if blen == 0 {
        return [alen as u8; M];
    };
    if alen == 0 {
        return [blen as u8; M];
    }
    assert!(N * 8 == M);

    // myers-style algo
    let mut diags = vec![U8::<N>::splat(0); blen];
    //
    //
    let mut row_hp = vec![U8::<N>::splat(255); blen];
    let mut row_hn = vec![U8::<N>::splat(0); blen];

    let mut start_penalty = 0u8;
    let mut last_lo = 1000;

    for i in 0..alen {
        let mut curr_dnp_j;

        let c_a: [U8<N>; 8] = std::array::from_fn(|shift| {
            U8::<N>::from_array(std::array::from_fn(|s| a[shift + 8 * s][i] as u8))
        });

        let lo = i.saturating_sub(max_dist);
        let hi = (i + max_dist + 1).min(blen);
        if lo == 0 {
            start_penalty = i as u8;
        }

        let mut prev_hn_jm1;
        let mut prev_hp_jm1;
        let mut prev_hp_j;
        let mut prev_hn_j;

        last_lo = lo;
        {
            let j = lo;
            prev_hn_j = row_hn[j];
            prev_hp_j = row_hp[j];

            let b_j = U8::<N>::splat(b[j]);
            let mut is_match = U8::<N>::splat(0);
            for shift in 0..8 {
                is_match |= c_a[shift]
                    .simd_eq(b_j)
                    .select(U8::<N>::splat(1 << shift), U8::<N>::splat(0));
            }

            // curr_d[j], before we used previous variable
            curr_dnp_j = prev_hn_j | is_match;

            // curr_h[j] = curr_d[j] - curr_v[j]
            // res := curr_dp[j] - curr_vp[j] + curr_vn[j];
            row_hp[j] = U8::<N>::splat(0); // res > 0
            row_hn[j] = curr_dnp_j; // res < 0

            diags[j] = !curr_dnp_j;
        }

        for j in lo + 1..hi {
            prev_hn_jm1 = prev_hn_j;
            prev_hp_jm1 = prev_hp_j;
            prev_hn_j = row_hn[j];
            prev_hp_j = row_hp[j];

            let b_j = U8::<N>::splat(b[j]);
            let mut is_match = U8::<N>::splat(0);
            for shift in 0..8 {
                is_match |= c_a[shift]
                    .simd_eq(b_j)
                    .select(U8::<N>::splat(1 << shift), U8::<N>::splat(0));
            }
            // curr_d[j - i] = prev_h[ j - 1 ] + curr_v [ j ]
            // res := curr_dp[j - 1] - prev_hp[j - 1] + prev_hn[j - 1];
            let curr_dnp_j_m1 = curr_dnp_j;
            let curr_vp_j = prev_hn_jm1 | !(curr_dnp_j_m1 | prev_hp_jm1); // res > 0
            let curr_vn_j = prev_hp_jm1 & curr_dnp_j_m1; // res < 0

            // curr_d[j], before we used previous variable
            curr_dnp_j = prev_hn_j | curr_vn_j | is_match;

            // curr_h[j] = curr_d[j] - curr_v[j]
            // res := curr_dp[j] - curr_vp[j] + curr_vn[j];
            row_hp[j] = !(curr_dnp_j | curr_vp_j) | curr_vn_j; // res > 0
            row_hn[j] = curr_vp_j & curr_dnp_j; // res < 0
        }
    }

    let one = U8::<N>::splat(1);
    let mut result = [0u8; M];
    let maxval = (max_dist + 1) as u8;

    for shift in 0..8 {
        let result_for_shift: U8<N> = U8::<N>::splat(start_penalty)
            + diags[..last_lo + 1]
                .iter()
                .map(|x| (*x >> shift) & one)
                .sum::<U8<N>>()
            + row_hp[last_lo + 1..]
                .iter()
                .map(|x| (*x >> shift) & one)
                .sum::<U8<N>>()
            - row_hn[last_lo + 1..]
                .iter()
                .map(|x| (*x >> shift) & one)
                .sum::<U8<N>>();

        for s in 0..N {
            result[shift as usize + s * 8] = result_for_shift[s].min(maxval);
        }
    }

    result
}

fn levenshtein_simd_by_1<const N: usize>(a: &[&[u8]; N], b: &[u8]) -> [u8; N] {
    // assumes all lengths are identical, not checked right now
    let (alen, blen) = (a[0].len(), b.len());
    let mut prev: Vec<U8<N>> = (0..=blen).map(|i| U8::<N>::splat(i as u8)).collect();
    let mut curr = vec![U8::<N>::splat(0); blen + 1];
    let one = U8::<N>::splat(1);

    for i in 1..=alen {
        curr[0] = U8::<N>::splat(i as u8);

        let c_a = U8::<N>::from_array(std::array::from_fn(|s| a[s][i - 1] as u8));

        for j in 1..=blen {
            let mask = c_a.simd_eq(U8::<N>::splat(b[j - 1]));

            curr[j] = (mask.select(prev[j - 1], prev[j - 1] + one))
                .simd_min(prev[j] + one)
                .simd_min(curr[j - 1] + one)
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    return *prev[blen].as_array();
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

pub fn bitty_levenshtein_n_by_1(a: Vec<&[u8]>, b: &[u8]) -> Vec<u8> {
    assert!(!b.contains(&255));
    a.chunks(128)
        .flat_map(|chunk| {
            if chunk.len() == 128 {
                bitty_levenshtein_simd_by_1::<16, 128>(chunk.try_into().unwrap(), b).to_vec()
            } else {
                panic!("chunk len is not 128");
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_SEQS: [&'static [u8]; 67] = [
        b"a", b"aa", b"aaa", b"aaaa", b"ab", b"aab", b"bab", b"abab", b"baba", b"Z9", b"k3x", b"T",
        b"q7", b"mN2", b"r", b"8b", b"L0p", b"dx", b"Y", b"w4R", b"3", b"tK", b"p9q", b"H2", b"s",
        b"Vx1", b"7", b"nB", b"c4", b"J", b"u8m", b"5t", b"g", b"R2d", b"y", b"0", b"eL", b"K9",
        b"z3Q", b"b", b"M1", b"f8", b"X", b"h2k", b"6", b"dP", b"q", b"9z", b"W4", b"l", b"C7r",
        b"2", b"vN", b"t", b"8Kx", b"G", b"m5", b"p", b"1aZ", b"r4", b"S", b"y7", b"k", b"D3",
        b"0x", b"n", b"B8q",
    ];

    #[test]
    fn test_bitty_0_0() {
        assert_eq!(_bitty_levenshtein(b"", b""), 0);
    }

    #[test]
    fn test_bitty_0_1() {
        assert_eq!(_bitty_levenshtein(b"", b"a"), 1);
        assert_eq!(_bitty_levenshtein(b"a", b""), 1);
    }

    #[test]
    fn test_bitty_1_1() {
        assert_eq!(_bitty_levenshtein(b"a", b"a"), 0);
        assert_eq!(_bitty_levenshtein(b"a", b"b"), 1);
    }

    #[test]
    fn test_bitty_1_2() {
        assert_eq!(_bitty_levenshtein(b"a", b"ab"), 1);
        assert_eq!(_bitty_levenshtein(b"a", b"bc"), 2);
    }

    #[test]
    fn test_bitty_2_1() {
        assert_eq!(_bitty_levenshtein(b"ab", b"a"), 1);
        assert_eq!(_bitty_levenshtein(b"bc", b"a"), 2);
    }

    #[test]
    fn test_bitty_2_2() {
        assert_eq!(_bitty_levenshtein(b"ab", b"ab"), 0);
        assert_eq!(_bitty_levenshtein(b"ab", b"ac"), 1);
        assert_eq!(_bitty_levenshtein(b"ab", b"bc"), 2);
        assert_eq!(_bitty_levenshtein(b"ab", b"cd"), 2);
    }

    #[test]
    fn stress_test_bitty() {
        for a in TEST_SEQS.iter() {
            for b in TEST_SEQS.iter() {
                let a_str = std::str::from_utf8(a).unwrap();
                let b_str = std::str::from_utf8(b).unwrap();
                let bitwise_result = _bitty_levenshtein(a, b) as i32;
                let reference = _levenshtein(a_str, b_str) as i32;
                assert_eq!(bitwise_result, reference);
            }
        }
    }

    #[test]
    fn stress_test_bitty_simd_1_to_n() {
        let consts: [&[u8]; 5] = [b"", b" ", b"  ", b"   ", b"    "];
        for a in TEST_SEQS.iter() {
            for b in TEST_SEQS.iter() {
                let a_str = std::str::from_utf8(a).unwrap();
                let b_str = std::str::from_utf8(b).unwrap();
                let input: [&[u8]; 256] =
                    std::array::from_fn(|s| if s % 13 != 0 { *a } else { consts[(*a).len()] });

                let bitwise_results = bitty_levenshtein_simd_by_1::<32, 256>(&input, b);
                for (i, res) in bitwise_results.into_iter().enumerate() {
                    let reference =
                        _levenshtein(std::str::from_utf8(input[i]).unwrap(), b_str) as i32;
                    assert_eq!(res as i32, reference, "{} {} {}", a_str, b_str, i);
                }
            }
        }
    }

    #[test]
    fn stress_test_bitty_simd_1_to_n_limited() {
        let test_seqs_long: [&[u8]; 11] = [
            b"",
            b" ",
            b"  ",
            b"   ",
            b"    ",
            b"abab",
            b"bab",
            b"baba",
            b"abababab",
            b"babababa",
            b"babbbbaba",
        ];
        for a in test_seqs_long.iter() {
            for b in test_seqs_long.iter() {
                for max_dist in 1..10i32 {
                    let a_str = std::str::from_utf8(a).unwrap();
                    let b_str = std::str::from_utf8(b).unwrap();
                    let input: [&[u8]; 256] = std::array::from_fn(|_s| *a);

                    let bitwise_results = bitty_levenshtein_simd_by_1_limited::<32, 256>(
                        &input,
                        b,
                        max_dist as usize,
                    );

                    let result = bitwise_results[0] as i32;
                    let reference_uncut =
                        _levenshtein(std::str::from_utf8(input[0]).unwrap(), b_str) as i32;
                    let reference = reference_uncut.min(max_dist + 1);
                    assert_eq!(result, reference, "|{}| |{}| {}", a_str, b_str, max_dist);
                }
            }
        }
    }
}
