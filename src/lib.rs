#![feature(portable_simd)]
#![feature(generic_const_exprs)]

#[cfg(feature = "python")]
use {
    crate::simd as levenshtein_simd,
    pyo3::prelude::*,
    pyo3::types::{PyBytes, PyList},
};

pub mod simd;

/// Compute the Levenshtein distance between two byte slices using a single-row DP approach.
pub fn levenshtein_bytes(a: &[u8], b: &[u8]) -> usize {
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
#[cfg(feature = "python")]
#[pyfunction]
fn compute_levenshtein_1_to_n_bytes(left: &[u8], right: Vec<Vec<u8>>) -> Vec<usize> {
    right
        .iter()
        .map(|r| levenshtein_bytes(left, r.as_slice()))
        .collect()
}

/// Compute Levenshtein distances from one byte slice to a list of byte slices using SIMD.
#[cfg(feature = "python")]
#[pyfunction]
fn compute_levenshtein_1_to_n_bytes_simd(left: &[u8], right: Vec<Vec<u8>>) -> Vec<usize> {
    let results =
        levenshtein_simd::levenshtein_n_by_1(right.iter().map(|r| r.as_slice()).collect(), left);
    results.iter().map(|x| *x as usize).collect()
}

/// Compute Levenshtein distances from one byte slice to a list of byte slices using Bitty SIMD.
#[cfg(feature = "python")]
#[pyfunction]
fn compute_levenshtein_1_to_n_bytes_bitty_simd<'py>(
    _py: Python<'py>,
    left: &[u8],
    right: Bound<'py, PyList>,
) -> PyResult<Vec<usize>> {
    let rights_tmp: Vec<Bound<'py, PyBytes>> = right
        .iter()
        .map(|r| r.downcast_into::<PyBytes>().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;
    let rights = rights_tmp.iter().map(|b| b.as_bytes()).collect();

    let results = levenshtein_simd::bitty_levenshtein_n_by_1(&rights, left);
    Ok(results.iter().map(|x| *x as usize).collect())
}

/// Compute Levenshtein distances from each byte slice in `left` to each byte slice in `right`.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (left, right, max_dist=254))]
fn compute_levenshtein_m_to_n<'py>(
    _py: Python<'py>,
    left: Bound<'py, PyList>,
    right: Bound<'py, PyList>,
    max_dist: i32,
) -> PyResult<Vec<Vec<usize>>> {
    const CHUNK_SIZE: usize = 256;
    assert!(max_dist <= 254, "max_dist must be less than 255");

    let left_bounds: Vec<Bound<'py, PyBytes>> = left
        .iter()
        .map(|x| x.downcast_into::<PyBytes>().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;
    let right_bounds: Vec<Bound<'py, PyBytes>> = right
        .iter()
        .map(|x| x.downcast_into::<PyBytes>().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;

    let lefts: Vec<&[u8]> = left_bounds.iter().map(|b| b.as_bytes()).collect();
    let rights: Vec<&[u8]> = right_bounds.iter().map(|b| b.as_bytes()).collect();

    let mut indices: Vec<usize> = (0..lefts.len()).collect();
    indices.sort_by_key(|&idx| lefts[idx].len());

    let mut result_indexed = vec![vec![]; lefts.len()];
    let padding_slice: &[u8] = b"";

    let mut i = 0;
    while i < indices.len() {
        let current_len = lefts[indices[i]].len();
        let mut j = i;
        while j < indices.len() && lefts[indices[j]].len() == current_len {
            j += 1;
        }

        let same_len_indices = &indices[i..j];
        for chunk in same_len_indices.chunks(CHUNK_SIZE) {
            let chunk_len = chunk.len();
            let mut chunk_arr = [&[] as &[u8]; CHUNK_SIZE];
            for (k, &idx) in chunk.iter().enumerate() {
                chunk_arr[k] = lefts[idx];
            }
            for k in chunk_len..CHUNK_SIZE {
                chunk_arr[k] = padding_slice;
            }

            let chunk_res = levenshtein_simd::bitty_levenshtein_simd_by_n_limited::<CHUNK_SIZE>(
                &chunk_arr,
                &rights,
                max_dist as usize,
            );

            for k in 0..chunk_len {
                result_indexed[chunk[k]] =
                    chunk_res[k].iter().map(|&x| x as usize).collect::<Vec<_>>();
            }
        }
        i = j;
    }

    Ok(result_indexed)
}

#[cfg(feature = "python")]
#[pymodule]
fn rust_levenshtein(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_levenshtein_1_to_n_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(compute_levenshtein_1_to_n_bytes_simd, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_levenshtein_1_to_n_bytes_bitty_simd,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compute_levenshtein_m_to_n, m)?)?;
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
