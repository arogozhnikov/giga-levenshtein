TODOs:

- 1_by_n : control for different lengths
- wrapper for 1 by n should immediately remove sequences with significant distance mismatch
- m by n: wrapper should group targets by length (worth transposing if helpful?)
  - then running only against targets with lengths in the correct bucket.
- test parity of results against python_levenshtein for random 256 x 256 strings


- x] m by n: should pre-compute present symbols.
