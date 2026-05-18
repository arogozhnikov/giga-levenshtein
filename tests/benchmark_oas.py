from typing import Counter
from pathlib import Path
import random

import time
from dataclasses import dataclass

import giga_levenshtein

import Levenshtein
import gzip
from functools import lru_cache

dist_func = Levenshtein.distance


def py_m_to_n(left: list[bytes], right: list[bytes]) -> list[list[int]]:
    return [[dist_func(l, r) for r in right] for l in left]


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


@lru_cache()
def get_oas_sequences() -> list[bytes]:
    with gzip.open(
        Path(__file__).parent.parent / "benches/heavy_seqs.txt.gz", "rb"
    ) as f:
        result = [line.strip()[:400] + b"a" * 100 for line in f]
        assert all(len(x) == 500 for x in result), Counter([len(x) for x in result])
        result = result[:256]
        return result


def vary(x: list[bytes]) -> list[bytes]:
    # varying right in bench_m_to_n has little effect,
    # while varying left shows effect below

    result = x[:]
    # different options for inputs were tried here
    # option 1: no changes - fast, 35ms
    # option 2: reduce size of some entries, 20ms
    # for i in range(len(result)):
    #     if i % 2 == 0:
    #         result[i] = b""
    # option 3: double sequences, 109 ms
    # for i in range(len(result)):
    #     result[i] = result[i] + result[i]
    # option 4: all different lengths, 10241 ms
    # for i in range(len(result)):
    #     result[i] = result[i] + result[i][:i]
    # option 5: add just small variability in lengths, 1113ms
    # for i in range(len(result)):
    #     result[i] = result[i] + result[i][: i % 32]
    # option 6: double half of sequences, 90ms
    # for i in range(len(result)):
    #     if i % 2 == 0:
    #         result[i] = result[i] + result[i]

    return result


def bench_m_to_n(dist: int) -> BenchResult:
    left = vary(get_oas_sequences())
    right = get_oas_sequences()
    py_ms = _timeit(py_m_to_n, left, right)
    rs_ms = _timeit(giga_levenshtein.compute_levenshtein_m_to_n, left, right, dist)
    return BenchResult(f"{len(left):>3}_to_{len(right):>3}_{dist}", py_ms, rs_ms)


def main():
    random.seed(42)
    results: list[BenchResult] = []

    print("\n### compute_levenshtein_m_to_n ###\n")
    for dist in [8, 32]:
        results.append(bench_m_to_n(dist))
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
