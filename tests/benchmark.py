import random

import time
from dataclasses import dataclass

import giga_levenshtein

import Levenshtein

dist_func = Levenshtein.distance


def py_1_to_n(left: bytes, right: list[str]) -> list[int]:
    return [dist_func(left, r) for r in right]


def py_m_to_n(left: list[str], right: list[str]) -> list[list[int]]:
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


def bench_1_to_n_bitty_simd(n: int, str_len: int) -> BenchResult:
    left = random_bytes(str_len)
    # n must be a multiple of 128 for bitty_simd
    n = (n // -256) * -256
    right = [random_bytes(str_len) for _ in range(n)]

    py_ms = _timeit(py_1_to_n, left, right)
    rs_ms = _timeit(giga_levenshtein.compute_levenshtein_1_to_n, left, right)
    return BenchResult(f"1_to_{n}  (strlen={str_len})", py_ms, rs_ms)


def bench_m_to_n(m: int, n: int, str_len: int) -> BenchResult:
    left = [random_bytes(str_len) for _ in range(m)]
    right = [random_bytes(str_len) for _ in range(n)]

    py_ms = _timeit(py_m_to_n, left, right)
    rs_ms = _timeit(giga_levenshtein.compute_levenshtein_m_to_n, left, right)
    return BenchResult(f"{m}_to_{n}  (strlen={str_len})", py_ms, rs_ms)


def main(
    sizes: list[int] = [128, 256],
    str_lens: list[int] = [16, 64, 256, 1024],
):
    random.seed(42)
    results: list[BenchResult] = []

    print("=" * 90)
    print("giga_levenshtein benchmarks")
    print("=" * 90)

    print("\n### compute_levenshtein_1_to_n ###\n")
    # bitty_simd internally uses chunks of 256 for now
    bitty_sizes = [256]
    for n in bitty_sizes:
        for sl in str_lens:
            r = bench_1_to_n_bitty_simd(n, sl)
            results.append(r)
            print(r)

    print("\n### compute_levenshtein_m_to_n ###\n")
    for n in sizes:
        for sl in str_lens:
            r = bench_m_to_n(n, n, sl)
            results.append(r)
            print(r)


if __name__ == "__main__":
    main()
