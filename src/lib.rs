#![feature(portable_simd)]
use pyo3::prelude::*;

mod simd;
use crate::simd as levenshtein_simd;

/// Compute the Levenshtein distance between two byte slices using a single-row DP approach.
fn levenshtein_bytes(a: &[u8], b: &[u8]) -> usize {
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
        return levenshtein_bytes(b, a);
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

/// Compute Levenshtein distances from one byte slice to a list of byte slices.
#[pyfunction]
fn compute_levenshtein_1_to_n_bytes(left: &[u8], right: Vec<Vec<u8>>) -> Vec<usize> {
    right
        .iter()
        .map(|r| levenshtein_bytes(left, r.as_slice()))
        .collect()
}

/// Compute Levenshtein distances from one byte slice to a list of byte slices using SIMD.
#[pyfunction]
fn compute_levenshtein_1_to_n_bytes_simd(left: &[u8], right: Vec<Vec<u8>>) -> Vec<usize> {
    let results =
        levenshtein_simd::levenshtein_n_by_1(right.iter().map(|r| r.as_slice()).collect(), left);
    results.iter().map(|x| *x as usize).collect()
}

/// Compute Levenshtein distances from one byte slice to a list of byte slices using Bitty SIMD.
#[pyfunction]
fn compute_levenshtein_1_to_n_bytes_bitty_simd(left: &[u8], right: Vec<Vec<u8>>) -> Vec<usize> {
    let results = levenshtein_simd::bitty_levenshtein_n_by_1(
        right.iter().map(|r| r.as_slice()).collect(),
        left,
    );
    results.iter().map(|x| *x as usize).collect()
}

/// Compute Levenshtein distances from each byte slice in `left` to each byte slice in `right`.
#[pyfunction]
fn compute_levenshtein_m_to_n_bytes(left: Vec<Vec<u8>>, right: Vec<Vec<u8>>) -> Vec<Vec<usize>> {
    left.iter()
        .map(|l| {
            right
                .iter()
                .map(|r| levenshtein_bytes(l.as_slice(), r.as_slice()))
                .collect()
        })
        .collect()
}

#[pymodule]
fn rust_levenshtein(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_levenshtein_1_to_n_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(compute_levenshtein_1_to_n_bytes_simd, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_levenshtein_1_to_n_bytes_bitty_simd,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compute_levenshtein_m_to_n_bytes, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical() {
        assert_eq!(levenshtein_bytes("hello".as_bytes(), "hello".as_bytes()), 0);
    }

    #[test]
    fn test_empty() {
        assert_eq!(levenshtein_bytes("".as_bytes(), "".as_bytes()), 0);
        assert_eq!(levenshtein_bytes("abc".as_bytes(), "".as_bytes()), 3);
        assert_eq!(levenshtein_bytes("".as_bytes(), "abc".as_bytes()), 3);
    }

    #[test]
    fn test_basic() {
        assert_eq!(
            levenshtein_bytes("kitten".as_bytes(), "sitting".as_bytes()),
            3
        );
        assert_eq!(
            levenshtein_bytes("saturday".as_bytes(), "sunday".as_bytes()),
            3
        );
    }

    #[test]
    fn test_single_char() {
        assert_eq!(levenshtein_bytes("a".as_bytes(), "b".as_bytes()), 1);
        assert_eq!(levenshtein_bytes("a".as_bytes(), "a".as_bytes()), 0);
        assert_eq!(levenshtein_bytes("a".as_bytes(), "".as_bytes()), 1);
    }
}
