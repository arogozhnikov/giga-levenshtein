#![feature(portable_simd)]
#![feature(generic_const_exprs)]

#[cfg(feature = "python")]
use {
    crate::simd as levenshtein_simd,
    pyo3::prelude::*,
    pyo3::types::{PyBytes, PyList},
};

pub mod simd;

/// Compute Levenshtein distances from a single byte-slice to a list of byte-slices.
/// result[i] = levenshtein(left, right[i])
/// Exposing this implementation for benchmarking
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (left, right, max_dist=254))]
fn _compute_levenshtein_1_to_n_u64<'py>(
    _py: Python<'py>,
    left: &[u8],
    right: Bound<'py, PyList>,
    max_dist: i32,
) -> PyResult<Vec<usize>> {
    let rights_tmp: Vec<Bound<'py, PyBytes>> = right
        .iter()
        .map(|r| r.downcast_into::<PyBytes>().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;
    let rights: Vec<&[u8]> = rights_tmp
        .iter()
        .map(|b| b.as_bytes())
        .collect::<Vec<&[u8]>>();

    let results =
        levenshtein_simd::bitty_levenshtein_simd_by_1_limited_u64(&rights, left, max_dist as usize);
    Ok(results.iter().map(|x| *x as usize).collect())
}

/// Compute Levenshtein distances from a single byte-slice to a list of byte-slices.
/// result[i] = levenshtein(left, right[i])
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (left, right, max_dist=254))]
fn compute_levenshtein_1_to_n<'py>(
    _py: Python<'py>,
    left: &[u8],
    right: Bound<'py, PyList>,
    max_dist: i32,
) -> PyResult<Vec<usize>> {
    let rights_tmp: Vec<Bound<'py, PyBytes>> = right
        .iter()
        .map(|r| r.downcast_into::<PyBytes>().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;
    let rights: [&[u8]; 256] = rights_tmp
        .iter()
        .map(|b| b.as_bytes())
        .collect::<Vec<&[u8]>>()
        .try_into()
        .unwrap();

    let results =
        levenshtein_simd::bitty_levenshtein_simd_by_1_limited(&rights, left, max_dist as usize);
    Ok(results.iter().map(|x| *x as usize).collect())
}

/// Compute Levenshtein distances from each byte slice in `left` to each byte slice in `right`.
/// result[i][j] = levenshtein(left[i], right[j])
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

    let mut result = vec![vec![]; lefts.len()];
    let mut left_order: Vec<usize> = (0..lefts.len()).collect();
    left_order.sort_unstable_by_key(|&idx| lefts[idx].len());

    for chunk in left_order.chunks(CHUNK_SIZE) {
        let mut chunk_arr = [&[] as &[u8]; CHUNK_SIZE];
        for (k, &left_idx) in chunk.iter().enumerate() {
            chunk_arr[k] = lefts[left_idx];
        }

        let chunk_res = levenshtein_simd::bitty_levenshtein_simd_by_n_limited::<CHUNK_SIZE>(
            &chunk_arr,
            &rights,
            max_dist as usize,
        );

        for (k, &left_idx) in chunk.iter().enumerate() {
            result[left_idx] = chunk_res[k].iter().map(|&x| x as usize).collect::<Vec<_>>();
        }
    }

    Ok(result)
}

#[cfg(feature = "python")]
#[pymodule]
fn giga_levenshtein(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_compute_levenshtein_1_to_n_u64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_levenshtein_1_to_n, m)?)?;
    m.add_function(wrap_pyfunction!(compute_levenshtein_m_to_n, m)?)?;
    Ok(())
}
