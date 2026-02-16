from typing import Literal, Callable
import random
import string
import time
from dataclasses import dataclass

import rust_levenshtein


def _naive_py_levenshtein(a: bytes, b: bytes) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def py_1_to_n(
    left: bytes, right: list[str], dist_func: Callable[[str, str], int]
) -> list[int]:
    return [dist_func(left, r) for r in right]


def py_m_to_n(
    left: list[str], right: list[str], dist_func: Callable[[str, str], int]
) -> list[list[int]]:
    return [[dist_func(l, r) for r in right] for l in left]


def random_bytes(length: int) -> bytes:
    return bytes(random.choices(range(128), k=length))


@dataclass
class BenchResult:
    label: bytes
    python_ms: float
    rust_ms: float

    def __str__(self) -> str:
        speedup = self.python_ms / self.rust_ms if self.rust_ms > 0 else float("inf")
        return (
            f"  {self.label:<40s} "
            f"Python: {self.python_ms:>9.2f} ms | "
            f"Rust:   {self.rust_ms:>9.2f} ms | "
            f"Speedup: {speedup:>7.1f}x"
        )


def _timeit(fn, *args, repeats: int = 3) -> float:
    """Return the best-of-N wall time in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        elapsed = (time.perf_counter() - t0) * 1000
        best = min(best, elapsed)
    return best


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_1_to_n(
    n: int, str_len: int, dist_func: Callable[[str, str], int]
) -> BenchResult:
    left = random_bytes(str_len)
    right = [random_bytes(str_len) for _ in range(n)]

    py_ms = _timeit(py_1_to_n, left, right, dist_func)
    rs_ms = _timeit(rust_levenshtein.compute_levenshtein_1_to_n_bytes, left, right)
    return BenchResult(f"1_to_{n}  (strlen={str_len})", py_ms, rs_ms)


def bench_m_to_n(
    m: int, n: int, str_len: int, dist_func: Callable[[str, str], int]
) -> BenchResult:
    left = [random_bytes(str_len) for _ in range(m)]
    right = [random_bytes(str_len) for _ in range(n)]

    py_ms = _timeit(py_m_to_n, left, right, dist_func)
    rs_ms = _timeit(rust_levenshtein.compute_levenshtein_m_to_n_bytes, left, right)
    return BenchResult(f"{m}_to_{n}  (strlen={str_len})", py_ms, rs_ms)


def main(
    sizes: list[int] = [10, 30],
    str_lens: list[int] = [16, 64, 256],
    baseline: Literal["plain_python", "python_levenshtein"] = "python_levenshtein",
):
    if baseline == "plain_python":
        dist_func = _naive_py_levenshtein
    else:
        import Levenshtein

        dist_func = Levenshtein.distance

    random.seed(42)
    results: list[BenchResult] = []

    print("=" * 90)
    print(f" rust_levenshtein benchmarks (baseline: {baseline})")
    print("=" * 90)

    # --- 1-to-N ---
    print("\n### compute_levenshtein_1_to_n ###\n")
    for n in sizes:
        for sl in str_lens:
            r = bench_1_to_n(n, sl, dist_func)
            results.append(r)
            print(r)

    # --- M-to-N ---
    print("\n### compute_levenshtein_m_to_n ###\n")
    for n in sizes:
        for sl in str_lens:
            r = bench_m_to_n(n, n, sl, dist_func)
            results.append(r)
            print(r)


if __name__ == "__main__":
    main()
