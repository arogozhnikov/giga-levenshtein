## Very fast Levenshtein distance

Is it possible to speed up computation for Levenshtein distances in 2026? 

At a first thought, the answer is "likely no". 
Getting speed-ups was simple 30-40 years ago, but since then all kinds of tricks were invented:

- Ukkonen's banding trick
- Myers' bit-vector algorithm
- Method of "Four russians"
- special structures: trees / Levenshtein automaton 


This implementation is going all-in on SIMD while still combining previous tricks. Core idea is to compute multiple Levenshtein distances in parallel &mdash; a use-case that I have quite often.

- ~ 100x faster than dynamic programming baseline in rust
- 5x-8x faster than `python_levenshtein` on computing massive all-to-all distances
  (don't take this lightly, `python_levenshtein` has impressive performance)

And we still use just 1 thread.


### Implementation details / dev notes

Implemented in (nightly) rust using `portable_simd` crate.


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


Some improvements I'd like to cover:

1. ~~better 'stride narrowing'~~
2. smart (at least length-aware) prefiltering
3. real-data benchmark
4. ~~uint64-only implementation for comparison; it is possible that LLVM can batch uint64 operations into vector instructions with performance comparable to portable_simd.~~
   - u64 version is surprisingly efficient, maybe I'll implement m-to-n version for it.
   - u64 shows better performance than u8x8 SIMD while supposedly should use same/faster instructions.
5. support switch between 128/256/512-but wide SIMD and u64



