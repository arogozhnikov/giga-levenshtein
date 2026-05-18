use std::simd::cmp::SimdPartialEq;
use std::simd::{Select, Simd};

fn _naive_levenshtein_1_on_1(a: &[u8], b: &[u8]) -> i32 {
    let n = b.len();
    let mut prev: Vec<i32> = (0..=n as i32).collect();
    let mut curr: Vec<i32> = vec![0; n + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i as i32 + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// bitty: myers-style algo
/// this is "simplified reference" for following implementations
fn _bitty_levenshtein_1_on_1(a: &[u8], b: &[u8]) -> i32 {
    if b.is_empty() {
        return a.len() as i32;
    };
    if a.is_empty() {
        return b.len() as i32;
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
    result as i32
}

type Bits<const N: usize> = Simd<u8, { N / 8 }>;

fn _bitty_levenshtein_simd_by_1<const M: usize>(a: &[&[u8]; M], b: &[u8]) -> [i32; M]
where
    [(); M / 8]:,
{
    // assumes all lengths are identical, does not check this right now
    let (alen, blen) = (a[0].len(), b.len());

    if blen == 0 {
        return [alen as i32; M];
    };
    if alen == 0 {
        return [blen as i32; M];
    }

    // myers-style algo
    let mut prev_hp = vec![Bits::<M>::splat(255); blen];
    let mut prev_hn = vec![Bits::<M>::splat(0); blen];
    //
    let mut curr_hp = vec![Bits::<M>::splat(255); blen];
    let mut curr_hn = vec![Bits::<M>::splat(0); blen];

    for i in 0..alen {
        let mut is_matches: [Bits<M>; 256] = [Bits::<M>::splat(0); 256];

        for s in 0..M {
            let c = a[s][i] as usize;
            is_matches[c][s / 8] |= 1u8 << (s % 8);
        }

        let mut curr_dnp_j = Bits::<M>::splat(255);

        for (j, cb) in b.iter().enumerate() {
            let is_match = is_matches[*cb as usize];
            let curr_vp_j: Bits<M>;
            let curr_vn_j: Bits<M>;
            if j == 0 {
                curr_vp_j = Bits::<M>::splat(255);
                curr_vn_j = Bits::<M>::splat(0);
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

    let mut result = [alen as i32; M];

    sum_masks(&curr_hp, &mut result, true);
    sum_masks(&curr_hn, &mut result, false);

    result
}

fn sum_masks_u64(masks: &[u64], result: &mut [i32; 64], add: bool) {
    let mut accum = [0u64; 8];
    for (i, mask) in masks.iter().enumerate() {
        for shift in 0..8u8 {
            accum[shift as usize] += (*mask >> shift) & 0x0101010101010101u64;
        }
        if i % 200 == 199 || i + 1 == masks.len() {
            for shift in 0..8 {
                for s in 0..8 {
                    if add {
                        result[shift + 8 * s] += ((accum[shift] >> (8 * s)) & 255) as i32;
                    } else {
                        result[shift + 8 * s] -= ((accum[shift] >> (8 * s)) & 255) as i32;
                    }
                }
                accum[shift] = 0;
            }
        }
    }
}

fn sum_masks<const M: usize>(masks: &[Bits<M>], result: &mut [i32; M], add: bool) {
    let mut accum = [Bits::<M>::splat(0); 8];
    for (i, mask) in masks.iter().enumerate() {
        for shift in 0..8u8 {
            accum[shift as usize] += (*mask >> shift) & Bits::<M>::splat(1);
        }
        if i % 200 == 199 || i + 1 == masks.len() {
            for shift in 0..8 {
                for s in 0..(M / 8) {
                    if add {
                        result[shift + 8 * s] += accum[shift][s] as i32;
                    } else {
                        result[shift + 8 * s] -= accum[shift][s] as i32;
                    }
                }
                accum[shift] = Bits::<M>::splat(0);
            }
        }
    }
}

fn _bitty_levenshtein_u64_by_1_limited(a: &[&[u8]; 64], b: &[u8], max_dist: usize) -> [i32; 64] {
    // similar to simd-by-1-limited, but does not use SIMD.
    // surprisingly fast and does not need nightly.
    let (alen, blen) = (a.iter().map(|x| x.len()).max().unwrap_or(0), b.len());
    assert!(max_dist <= 254);

    let maxval = (max_dist + 1) as i32;
    if blen == 0 {
        return std::array::from_fn(|i| (a[i].len() as i32).min(maxval));
    };
    if alen == 0 {
        return [(blen as i32).min(maxval); 64];
    }

    // myers-style algo
    let mut row_hp = vec![u64::MAX; blen];
    let mut row_hn = vec![0u64; blen];

    let pad_sizes: [usize; 64] = std::array::from_fn(|i| alen - a[i].len());

    for i in 0..alen {
        let mut mask_is_pad = 0u64;
        let mut is_matches: [u64; 256] = [0u64; 256];

        for s in 0..64 {
            if i < pad_sizes[s] {
                mask_is_pad |= 1u64 << s;
            } else {
                is_matches[a[s][i - pad_sizes[s]] as usize] |= 1u64 << s;
            }
        }

        let lo = (i as i32 + blen as i32 - alen as i32 - max_dist as i32).max(0) as usize;
        let hi = ((i as i32 + max_dist as i32 + 1 + blen as i32 - alen as i32) as usize).min(blen);

        let mut prev_hp_j = mask_is_pad;
        let mut prev_hn_j = !mask_is_pad;
        let mut curr_dz_j = u64::MAX;

        for j in lo..hi {
            let prev_hn_jm1 = prev_hn_j;
            let prev_hp_jm1 = prev_hp_j;
            prev_hn_j = row_hn[j];
            prev_hp_j = row_hp[j];

            let is_match = is_matches[b[j] as usize];

            // curr_d[j - i] = prev_h[ j - 1 ] + curr_v [ j ]
            // res := curr_dp[j - 1] - prev_hp[j - 1] + prev_hn[j - 1];
            let curr_dz_j_m1 = curr_dz_j;
            let curr_vp_j = prev_hn_jm1 | !(curr_dz_j_m1 | prev_hp_jm1); // res > 0
            let curr_vn_j = prev_hp_jm1 & curr_dz_j_m1; // res < 0

            // curr_d[j], before we used previous variable
            curr_dz_j = prev_hn_j | curr_vn_j | is_match;

            // curr_h[j] = curr_d[j] - curr_v[j]
            // res := curr_dp[j] - curr_vp[j] + curr_vn[j];
            row_hp[j] = !(curr_dz_j | curr_vp_j) | curr_vn_j; // res > 0
            row_hn[j] = curr_vp_j & curr_dz_j; // res < 0
        }
    }

    let mut result = [0i32; 64];

    sum_masks_u64(&row_hp, &mut result, true);
    sum_masks_u64(&row_hn, &mut result, false);
    for i in 0..64 {
        result[i] = (result[i] + a[i].len() as i32).min(maxval);
    }
    result
}

pub fn bitty_levenshtein_simd_by_1_limited<const M: usize>(
    a: &[&[u8]; M],
    b: &[u8],
    max_dist: usize,
) -> [i32; M]
where
    [(); M / 8]:,
{
    // assumes all lengths are identical, does not check this right now
    // remark: pre-computing is_match is slower if max_dist is smaller than dictionary
    let (alen, max_blen) = (a.iter().map(|x| x.len()).max().unwrap_or(0), b.len());
    assert!(max_dist <= 254);
    // assert!((alen + max_dist < blen) && (blen + max_dist < alen));

    let maxval = (max_dist + 1) as i32;
    if max_blen == 0 {
        return std::array::from_fn(|i| (a[i].len() as i32).min(maxval));
    };
    if alen == 0 {
        return [(max_blen as i32).min(maxval); M];
    }
    let pad_sizes: [usize; M] = std::array::from_fn(|i| alen - a[i].len());
    let padded_a: [&[u8]; M] = std::array::from_fn(|i| {
        let mut padded = vec![b' '; pad_sizes[i]];
        padded.extend_from_slice(a[i]);
        padded.leak() as &[u8]
    });

    // myers-style algo
    let mut row_hp = vec![Bits::<M>::splat(255); max_blen];
    let mut row_hn = vec![Bits::<M>::splat(0); max_blen];

    for i in 0..alen {
        let c_a: [Bits<M>; 8] = std::array::from_fn(|shift| {
            Bits::<M>::from_array(std::array::from_fn(|s| padded_a[shift + 8 * s][i]))
        });

        let mask_is_pad = {
            let mut mask = Bits::<M>::splat(0u8);
            for shift in 0..8 {
                mask |= Bits::<M>::from_array(std::array::from_fn(|s| {
                    if i < pad_sizes[shift + 8 * s] {
                        1u8 << shift
                    } else {
                        0u8
                    }
                }));
            }
            mask
        };

        let lo = (i as i32 + max_blen as i32 - alen as i32 - max_dist as i32).max(0) as usize;
        let hi = ((i as i32 + max_dist as i32 + 1 + max_blen as i32 - alen as i32) as usize)
            .min(max_blen);

        let mut prev_hp_j = mask_is_pad;
        let mut prev_hn_j = !mask_is_pad;
        let mut curr_dz_j = Bits::<M>::splat(255);

        for j in lo..hi {
            let prev_hn_jm1 = prev_hn_j;
            let prev_hp_jm1 = prev_hp_j;
            prev_hn_j = row_hn[j];
            prev_hp_j = row_hp[j];

            let b_j = Bits::<M>::splat(b[j]);
            let mut is_match = Bits::<M>::splat(0);
            for shift in 0..8 {
                is_match |= c_a[shift]
                    .simd_eq(b_j)
                    .select(Bits::<M>::splat(1 << shift), Bits::<M>::splat(0));
            }

            // curr_d[j - i] = prev_h[ j - 1 ] + curr_v [ j ]
            // res := curr_dp[j - 1] - prev_hp[j - 1] + prev_hn[j - 1];
            let curr_dz_j_m1 = curr_dz_j;
            let curr_vp_j = prev_hn_jm1 | !(curr_dz_j_m1 | prev_hp_jm1); // res > 0
            let curr_vn_j = prev_hp_jm1 & curr_dz_j_m1; // res < 0

            // curr_d[j], before we used previous variable
            curr_dz_j = prev_hn_j | curr_vn_j | is_match;

            // curr_h[j] = curr_d[j] - curr_v[j]
            // res := curr_dp[j] - curr_vp[j] + curr_vn[j];
            row_hp[j] = !(curr_dz_j | curr_vp_j) | curr_vn_j; // res > 0
            row_hn[j] = curr_vp_j & curr_dz_j; // res < 0
        }
    }

    let mut result = [0i32; M];

    sum_masks(&row_hp, &mut result, true);
    sum_masks(&row_hn, &mut result, false);
    for i in 0..M {
        result[i] = (result[i] + a[i].len() as i32).min(maxval);
    }
    result
}

pub fn bitty_levenshtein_64_by_n_limited(
    a: &[&[u8]; 64],
    b: &Vec<&[u8]>,
    max_dist: usize,
) -> Vec<Vec<i32>> {
    let alen = a.iter().map(|x| x.len()).max().unwrap_or(0);
    let blen = b.iter().map(|x| x.len()).max().unwrap_or(0);
    let pad_sizes: [usize; 64] = std::array::from_fn(|i| alen - a[i].len());

    let maxval = (max_dist + 1) as i32;
    if blen == 0 {
        let res = a
            .iter()
            .map(|x| vec![(x.len() as i32).min(maxval); b.len()])
            .collect();
        return res;
    }
    if alen == 0 {
        let lens: Vec<i32> = b.iter().map(|x| (x.len() as i32).min(maxval)).collect();
        let res = vec![lens; a.len()];
        return res;
    }

    // myers-style algo
    let mut rows_hp = vec![vec![u64::MAX; blen]; b.len()];
    let mut rows_hn = vec![vec![0u64; blen]; b.len()];

    for i in 0..alen {
        let mut is_matches = [0u64; 256];
        let mut mask_is_pad = 0u64;

        for s in 0..64 {
            if i < pad_sizes[s] {
                mask_is_pad |= 1u64 << s;
            } else {
                is_matches[a[s][i - pad_sizes[s]] as usize] |= 1u64 << s;
            }
        }

        for (bseq_id, bseq) in b.iter().enumerate() {
            let row_hp = &mut rows_hp[bseq_id];
            let row_hn = &mut rows_hn[bseq_id];

            let mut prev_hp_j = mask_is_pad;
            let mut prev_hn_j = !mask_is_pad;
            let mut curr_dz_j = u64::MAX;

            let lo = (i as i32 + bseq.len() as i32 - alen as i32 - max_dist as i32).max(0) as usize;
            let hi = ((i as i32 + max_dist as i32 + 1 + bseq.len() as i32 - alen as i32) as usize)
                .min(bseq.len());

            for j in lo..hi {
                let prev_hp_jm1 = prev_hp_j;
                let prev_hn_jm1 = prev_hn_j;

                prev_hp_j = row_hp[j];
                prev_hn_j = row_hn[j];

                let is_match = is_matches[bseq[j] as usize];
                // curr_d[j - i] = prev_h[ j - 1 ] + curr_v [ j ]
                // res := curr_dp[j - 1] - prev_hp[j - 1] + prev_hn[j - 1];
                let curr_dz_j_m1 = curr_dz_j;
                let curr_vp_j = prev_hn_jm1 | !(curr_dz_j_m1 | prev_hp_jm1); // res > 0
                let curr_vn_j = prev_hp_jm1 & curr_dz_j_m1; // res < 0

                // curr_d[j], before we used previous variable
                curr_dz_j = prev_hn_j | curr_vn_j | is_match;

                // curr_h[j] = curr_d[j] - curr_v[j]
                // res := curr_dp[j] - curr_vp[j] + curr_vn[j];
                row_hp[j] = !(curr_dz_j | curr_vp_j) | curr_vn_j; // res > 0
                row_hn[j] = curr_vp_j & curr_dz_j; // res < 0
            }
        }
    }

    let mut result = vec![vec![0i32; b.len()]; a.len()];

    let maxval = (max_dist + 1) as i32;

    for j in 0..b.len() {
        if b[j].is_empty() {
            for i in 0..a.len() {
                result[i][j] = (a[i].len() as i32).min(maxval);
            }
        } else {
            let mut result_j = [0i32; 64];

            sum_masks_u64(&rows_hp[j][..b[j].len()], &mut result_j, true);
            sum_masks_u64(&rows_hn[j][..b[j].len()], &mut result_j, false);

            for (i, &res) in result_j.iter().enumerate() {
                result[i][j] = (res + a[i].len() as i32).min(maxval);
            }
        }
    }

    result
}

pub fn bitty_levenshtein_simd_by_n_limited<const M: usize>(
    a: &[&[u8]; M],
    b: &Vec<&[u8]>,
    max_dist: usize,
) -> Vec<Vec<i32>>
where
    [(); M / 8]:,
{
    let alen = a.iter().map(|x| x.len()).max().unwrap_or(0);
    let blen = b.iter().map(|x| x.len()).max().unwrap_or(0);
    let pad_sizes: [usize; M] = std::array::from_fn(|i| alen - a[i].len());

    let maxval = (max_dist + 1) as i32;
    if blen == 0 {
        let res = a
            .iter()
            .map(|x| vec![(x.len() as i32).min(maxval); b.len()])
            .collect();
        return res;
    }
    if alen == 0 {
        let lens: Vec<i32> = b.iter().map(|x| (x.len() as i32).min(maxval)).collect();
        let res = vec![lens; a.len()];
        return res;
    }

    let enumed_bsecs = {
        // do not compute for sequences that are too short or too long
        let min_alen = a.iter().map(|x| x.len()).min().unwrap_or(0);
        let max_alen = a.iter().map(|x| x.len()).max().unwrap_or(0);
        b.iter()
            .enumerate()
            .filter(|(_, s)| s.len() <= max_alen + max_dist)
            .filter(|(_, s)| s.len() + max_dist >= min_alen)
            .map(|(i, s)| (i, *s))
            .collect::<Vec<(usize, &[u8])>>()
    };

    let present_chars = {
        let mut char_is_present_in_b = [false; 256];
        for bseq in b {
            for &c in *bseq {
                char_is_present_in_b[c as usize] = true;
            }
        }
        char_is_present_in_b
            .iter()
            .enumerate()
            .filter(|(_, &present)| present)
            .map(|(c, _)| c)
            .collect::<Vec<usize>>()
    };

    // myers-style algo
    let mut rows_hp = vec![vec![Bits::<M>::splat(255); blen]; b.len()];
    let mut rows_hn = vec![vec![Bits::<M>::splat(0); blen]; b.len()];

    for i in 0..alen {
        let mut is_matches = [Bits::<M>::splat(0); 256];

        for shift in 0..8 {
            let ca_shift = Bits::<M>::from_array(std::array::from_fn(|s| {
                let pad_size = pad_sizes[shift + 8 * s];
                if i < pad_size {
                    0
                } else {
                    a[shift + 8 * s][i - pad_size]
                }
            }));
            for &c in present_chars.iter() {
                is_matches[c] |= ca_shift
                    .simd_eq(Bits::<M>::splat(c as u8))
                    .select(Bits::<M>::splat(1 << shift), Bits::<M>::splat(0));
            }
        }

        let mask_is_pad = {
            let mut mask = Bits::<M>::splat(0u8);
            for shift in 0..8 {
                mask |= Bits::<M>::from_array(std::array::from_fn(|s| {
                    if i < pad_sizes[shift + 8 * s] {
                        1u8 << shift
                    } else {
                        0u8
                    }
                }));
            }
            mask
        };

        for &(bseq_id, bseq) in enumed_bsecs.iter() {
            let row_hp = &mut rows_hp[bseq_id];
            let row_hn = &mut rows_hn[bseq_id];

            let mut prev_hp_j = mask_is_pad;
            let mut prev_hn_j = !mask_is_pad;
            let mut curr_dz_j = Bits::<M>::splat(255);

            let a_center_to_end = alen as i32 - i as i32; // same as center to end
            let b_center = bseq.len() as i32 - a_center_to_end;

            let lo = (b_center - max_dist as i32).clamp(0, bseq.len() as i32) as usize;
            let hi = (b_center + max_dist as i32 + 1).clamp(0, bseq.len() as i32) as usize;

            for j in lo..hi {
                let prev_hp_jm1 = prev_hp_j;
                let prev_hn_jm1 = prev_hn_j;
                prev_hp_j = row_hp[j];
                prev_hn_j = row_hn[j];

                let is_match = is_matches[bseq[j] as usize];
                // curr_d[j - i] = prev_h[ j - 1 ] + curr_v [ j ]
                // res := curr_dp[j - 1] - prev_hp[j - 1] + prev_hn[j - 1];
                let curr_dz_j_m1 = curr_dz_j;
                let curr_vp_j = prev_hn_jm1 | !(curr_dz_j_m1 | prev_hp_jm1); // res > 0
                let curr_vn_j = prev_hp_jm1 & curr_dz_j_m1; // res < 0

                // curr_d[j], before we used previous variable
                curr_dz_j = prev_hn_j | curr_vn_j | is_match;

                // curr_h[j] = curr_d[j] - curr_v[j]
                // res := curr_dp[j] - curr_vp[j] + curr_vn[j];
                row_hp[j] = !(curr_dz_j | curr_vp_j) | curr_vn_j; // res > 0
                row_hn[j] = curr_vp_j & curr_dz_j; // res < 0
            }
        }
    }

    let mut result = vec![vec![maxval; b.len()]; a.len()];
    for &(j, bseq) in enumed_bsecs.iter() {
        if bseq.is_empty() {
            for i in 0..a.len() {
                result[i][j] = (a[i].len() as i32).min(maxval);
            }
        } else {
            let mut result_j = [0i32; M];

            sum_masks(&rows_hp[j][..bseq.len()], &mut result_j, true);
            sum_masks(&rows_hn[j][..bseq.len()], &mut result_j, false);

            for (i, &res) in result_j.iter().enumerate() {
                result[i][j] = (res + a[i].len() as i32).min(maxval);
            }
        }
    }

    result
}

pub fn bitty_levenshtein_n_by_1(a: &Vec<&[u8]>, b: &[u8]) -> Vec<i32> {
    assert!(!b.contains(&255));
    const CHUNK_SIZE: usize = 256;

    (*a).chunks(CHUNK_SIZE)
        .flat_map(|chunk| {
            if chunk.len() == CHUNK_SIZE {
                _bitty_levenshtein_simd_by_1::<CHUNK_SIZE>(chunk.try_into().unwrap(), b).to_vec()
            } else {
                panic!("chunk len is not {CHUNK_SIZE}",);
            }
        })
        .collect()
}

pub fn bitty_levenshtein_simd_by_1_limited_u64(
    a: &Vec<&[u8]>,
    b: &[u8],
    max_dist: usize,
) -> Vec<i32> {
    assert!(!b.contains(&255));
    const CHUNK_SIZE: usize = 64;

    (*a).chunks(CHUNK_SIZE)
        .flat_map(|chunk| {
            if chunk.len() == CHUNK_SIZE {
                _bitty_levenshtein_u64_by_1_limited(chunk.try_into().unwrap(), b, max_dist).to_vec()
            } else {
                panic!("chunk len is not {CHUNK_SIZE}",);
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    const SHORT_TEST_SEQS: [&[u8]; 70] = [
        b"", b"a", b"aa", b"aaa", b"aaaa", b"ab", b"ba", b"ac", b"aab", b"bab", b"abab", b"baba",
        b"Z9", b"k3x", b"T", b"q7", b"mN2", b"r", b"8b", b"L0p", b"dx", b"Y", b"w4R", b"3", b"tK",
        b"p9q", b"H2", b"s", b"Vx1", b"7", b"nB", b"c4", b"J", b"u8m", b"5t", b"g", b"R2d", b"y",
        b"0", b"eL", b"K9", b"z3Q", b"b", b"M1", b"f8", b"X", b"h2k", b"6", b"dP", b"q", b"9z",
        b"W4", b"l", b"C7r", b"2", b"vN", b"t", b"8Kx", b"G", b"m5", b"p", b"1aZ", b"r4", b"S",
        b"y7", b"k", b"D3", b"0x", b"n", b"B8q",
    ];

    const TEST_SEQS_LONG: [&[u8]; 17] = [
        b"",
        b" ",
        b"  ",
        b"   ",
        b"    ",
        b"ab",
        b"abc",
        b"cab",
        b"cababc",
        b"axc",
        b"ac",
        b"abab",
        b"bab",
        b"baba",
        b"abababab",
        b"babababa",
        b"babbbbaba",
    ];

    const PADS: [&[u8]; 256] = [b""; 256];

    fn get_many_sequences() -> Vec<&'static [u8]> {
        [
            TEST_SEQS_LONG.into_iter().collect::<Vec<&[u8]>>(),
            SHORT_TEST_SEQS.into_iter().collect::<Vec<&[u8]>>(),
            PADS.into_iter().collect::<Vec<&[u8]>>(),
        ]
        .concat()
        .into_iter()
        .take(256)
        .collect()
    }

    #[test]
    fn stress_test_bitty_1_on_1() {
        for a in SHORT_TEST_SEQS.iter() {
            for b in SHORT_TEST_SEQS.iter() {
                let bitwise_result = _bitty_levenshtein_1_on_1(a, b);
                let reference = _naive_levenshtein_1_on_1(a, b);
                assert_eq!(bitwise_result, reference);
            }
        }
    }

    #[test]
    fn stress_test_bitty_simd_by_1() {
        let consts: [&[u8]; 5] = [b"", b" ", b"  ", b"   ", b"    "];
        for a in SHORT_TEST_SEQS.iter() {
            for b in SHORT_TEST_SEQS.iter() {
                let input: [&[u8]; 256] =
                    std::array::from_fn(|s| if s % 13 != 0 { *a } else { consts[(*a).len()] });

                let bitwise_results = _bitty_levenshtein_simd_by_1::<256>(&input, b);
                for (i, res) in bitwise_results.into_iter().enumerate() {
                    let reference = _naive_levenshtein_1_on_1(input[i], b);
                    assert_eq!(res, reference, "{a:?} {b:?} {i}");
                }
            }
        }
    }

    #[test]
    fn stress_test_bitty_simd_by_1_limited() {
        for a in get_many_sequences() {
            for b in get_many_sequences() {
                for max_dist in 0..10i32 {
                    let input: [&[u8]; 256] = std::array::from_fn(|_s| a);

                    let bitwise_results =
                        bitty_levenshtein_simd_by_1_limited::<256>(&input, b, max_dist as usize);

                    let result = bitwise_results[0];
                    let reference_uncut = _naive_levenshtein_1_on_1(input[0], b);
                    let reference = reference_uncut.min(max_dist + 1);
                    assert_eq!(result, reference, "{a:?} {b:?} {max_dist}");
                }
            }
        }
    }

    #[test]
    fn stress_test_bitty_simd_by_1_limited_mixed_sizes() {
        let all_seqs = get_many_sequences();

        for b in all_seqs.iter() {
            for max_dist in 1..22i32 {
                let input: [&[u8]; 256] = std::array::from_fn(|s| all_seqs[s]);

                let bitwise_results =
                    bitty_levenshtein_simd_by_1_limited::<256>(&input, b, max_dist as usize);

                for (i, &res) in bitwise_results.iter().enumerate() {
                    let reference_uncut = _naive_levenshtein_1_on_1(input[i], b);
                    let reference = reference_uncut.min(max_dist + 1);

                    assert_eq!(
                        res,
                        reference,
                        "|{:?}| |{:?}| {} dist={max_dist}",
                        input[i],
                        b,
                        b.len()
                    );
                }
            }
        }
    }

    #[test]
    fn stress_test_bitty_u64_by_1_limited_mixed_sizes() {
        let all_seqs = get_many_sequences();
        let all_seqs_chunks: Vec<[&[u8]; 64]> = all_seqs
            .chunks(64)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();

        for b in all_seqs.iter() {
            for max_dist in 1..22i32 {
                for &input in all_seqs_chunks.iter() {
                    let bitwise_results =
                        _bitty_levenshtein_u64_by_1_limited(&input, b, max_dist as usize);

                    for (i, &res) in bitwise_results.iter().enumerate() {
                        let reference_uncut = _naive_levenshtein_1_on_1(input[i], b);
                        let reference = reference_uncut.min(max_dist + 1);

                        assert_eq!(
                            res,
                            reference,
                            "|{:?}| |{:?}| {} dist={max_dist}",
                            input[i],
                            b,
                            b.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stress_test_bitty_u64_by_n_limited() {
        let source_a_seqs = get_many_sequences();

        for max_aseqs in [1, 2, 4, 10, 20] {
            for n_bseqs in [1, 17, 302] {
                for max_blen in [0, 2, 4, 7, 20] {
                    for max_dist in 0..10i32 {
                        let n_a_seqs = max_aseqs.min(source_a_seqs.len());
                        let a_strs: [&[u8]; 64] =
                            std::array::from_fn(|i| source_a_seqs[i % n_a_seqs]);
                        // pad all a_strs to same length
                        let a_strs = if max_blen > 0 {
                            a_strs.map(|s| {
                                let mut padded = vec![0u8; max_blen];
                                let taken_len = max_blen.min(s.len());
                                padded[..taken_len].copy_from_slice(&s[..taken_len]);
                                padded.leak() as &[u8]
                            })
                        } else {
                            a_strs
                        };

                        let b_strs: Vec<&[u8]> = get_many_sequences()
                            .into_iter()
                            .cycle()
                            .take(n_bseqs)
                            .collect();

                        let bitwise_results =
                            bitty_levenshtein_64_by_n_limited(&a_strs, &b_strs, max_dist as usize);

                        for (i, res) in bitwise_results.iter().enumerate() {
                            for (j, &b_str) in b_strs.iter().enumerate() {
                                let reference_uncut = _naive_levenshtein_1_on_1(a_strs[i], b_str);
                                let reference = reference_uncut.min(max_dist + 1);

                                assert_eq!(
                                    (*res)[j],
                                    reference,
                                    "|{:?}| |{b_str:?}| {} dist={max_dist}",
                                    a_strs[i],
                                    b_str.len(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stress_test_bitty_simd_by_n_limited() {
        let source_a_seqs = get_many_sequences();

        for max_aseqs in [1, 2, 4, 10, 20] {
            for n_bseqs in [1, 17, 302] {
                for max_dist in 0..10i32 {
                    let n_a_seqs = max_aseqs.min(source_a_seqs.len());
                    let a_strs: [&[u8]; 256] = std::array::from_fn(|i| source_a_seqs[i % n_a_seqs]);

                    let b_strs: Vec<&[u8]> = get_many_sequences()
                        .into_iter()
                        .cycle()
                        .take(n_bseqs)
                        .collect();

                    let bitwise_results = bitty_levenshtein_simd_by_n_limited::<256>(
                        &a_strs,
                        &b_strs,
                        max_dist as usize,
                    );

                    for (res, &a_str) in bitwise_results.iter().zip(a_strs.iter()) {
                        for (j, &b_str) in b_strs.iter().enumerate() {
                            let reference_uncut = _naive_levenshtein_1_on_1(a_str, b_str);
                            let reference = reference_uncut.min(max_dist + 1);

                            assert_eq!(
                                (*res)[j],
                                reference,
                                "|{a_str:?}| |{b_str:?}| max_dist={max_dist}"
                            );
                        }
                    }
                }
            }
        }
    }
}
