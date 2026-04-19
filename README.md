## Very fast Levenshtein distance

Is it possible to speed up computation for Levenshtein distances? 

Getting speed-ups was simple 30-40 years ago, but since then all kinds of tricks were invented:

- Ukkonen's banding trick
- Myers' bit-vector algorithm
- Method of "Four russians"
- special structures: trees / Levenshtein automaton 


This implementation is going all-in on SIMD while still combining previous tricks.

- ~ 100x faster than dynamic programming baseline in rust
- 5x-8x faster than `python_levenshtein` on computing massive all-to-all distances
  (FYI `python_levenshtein` has impressive performance)

And we still use just 1 thread.


### Implementation details / dev notes

Implemented in (nightly) rust using portable_simd.


Tools:

- `cargo +nightly test`
- `cargo +nightly bench` - rust-side benchmarks
- `RUSTUP_TOOLCHAIN=nightly pip install -e . && python './tests/benchmark.py'` - benchmarks for python wrapper
- you may also need `pip install maturin`


In case of breaking changes (portable_simd is unstable, and API *does* change), I used 

```
rustup run nightly rustc --version
rustc 1.95.0-nightly (873b4beb0 2026-02-15)
```


Some TODOs:

1. better 'stride narrowing'
2. more flexible wrapper
3. support switch between 128/256/512-but wide SIMD
4. smart batching on length
5. prepare relevant benchmark
6. bench on AVX512 - I only tested on processors with 256-bit wide SIMD
7. uint64-only implementation for comparison; it is possible that LLVM can batch uint64 operations into vector instructions with performance comparable to portable_simd.



