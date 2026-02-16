use pyo3::prelude::*;

/// Compute the Levenshtein distance between two strings using a single-row DP approach.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Ensure we iterate over the shorter dimension for memory efficiency.
    if m < n {
        return levenshtein(b, a);
    }

    // `prev` holds the previous row of the DP matrix.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Compute Levenshtein distances from one string to a list of strings.
///
/// Returns a list of distances where the i-th element is the distance
/// between `left` and `right[i]`.
#[pyfunction]
fn compute_levenshtein_1_to_n(left: &str, right: Vec<String>) -> Vec<usize> {
    right.iter().map(|r| levenshtein(left, r)).collect()
}

/// Compute Levenshtein distances from each string in `left` to each string in `right`.
///
/// Returns a list of lists where the element at `[i][j]` is the distance
/// between `left[i]` and `right[j]`.
#[pyfunction]
fn compute_levenshtein_m_to_n(left: Vec<String>, right: Vec<String>) -> Vec<Vec<usize>> {
    // let res = levenshtein(&left[0], &right[0]);
    // left.iter()
    //     .map(|l| right.iter().map(|r| res).collect())
    //     .collect()

    left.iter()
        .map(|l| right.iter().map(|r| levenshtein(l, r)).collect())
        .collect()
}

#[pymodule]
fn rust_levenshtein(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_levenshtein_1_to_n, m)?)?;
    m.add_function(wrap_pyfunction!(compute_levenshtein_m_to_n, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical() {
        assert_eq!(levenshtein("hello", "hello"), 0);
    }

    #[test]
    fn test_empty() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn test_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("saturday", "sunday"), 3);
    }

    #[test]
    fn test_single_char() {
        assert_eq!(levenshtein("a", "b"), 1);
        assert_eq!(levenshtein("a", "a"), 0);
        assert_eq!(levenshtein("a", ""), 1);
    }
}
