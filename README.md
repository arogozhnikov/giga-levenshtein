## Very fast Levenshtein distance

It is really hard to speed up Levenshtein distance computation. 

It was simple 30-40 years ago, but since then all kinds of tricks were invented:

- Myers' bit-vector algorithm
- Ukkonen's banding trick
- "Four russians" lookups
- special structures: trees / Levenshtein automaton 


This implementation is going all-in on SIMD.

- ~ 100x faster than naive DP in rust (still using 1 thread)
- 5x-8x faster than `python_levenshtein` on computing massive all-to-all distances
  (`python_levenshtein` has impressive performance)



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
5. prepare relevant benchmark + test this on Intel's AVX512
6. uint64-only implementation for comparison; it is possible that proper simd-ification can be done automatically.



