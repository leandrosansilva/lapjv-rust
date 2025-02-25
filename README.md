# lapjv

[![Crates.io](https://img.shields.io/crates/v/lapjv.svg)](https://crates.io/crates/lapjv) [![Crates.io](https://img.shields.io/crates/d/lapjv.svg)](https://crates.io/crates/lapjv)
[![Build Status](https://travis-ci.org/Antti/lapjv-rust.svg?branch=master)](https://travis-ci.org/Antti/lapjv-rust)

## Linear Assignment Problem solver using Jonker-Volgenant algorithm


This is rust implementation of the Jonker-Volgenant algorithm for linear assignment problem

* [documentation](https://docs.rs/lapjv/)
* [website](https://github.com/Antti/lapjv/)

## Example usage:

```rust
use lapjv::LapJV;
use nalgebra::DMatrixView;

let mut solver = LapJV::default();
let m: SMatrix<f64, 3, 3> = SMatrix::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
let result = solver.solve(m.as_view()).unwrap();
assert_eq!(*result.0, vec![2, 0, 1]);
assert_eq!(*result.1, vec![1, 2, 0]);
```

You can call `solver.solve()` any arbitrary number of times, so that it will try to reuse the
same dynamic allocated memory across the usages, likely increasing performance.
