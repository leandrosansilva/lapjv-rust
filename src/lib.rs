use log::trace;
use nalgebra::{DMatrixView, DVectorView};
use num_traits::Float;

use std::fmt;
use std::ops::{AddAssign, SubAssign};

pub trait LapJVCost: Float + AddAssign + SubAssign + std::fmt::Debug {}
impl<T> LapJVCost for T where T: Float + AddAssign + SubAssign + std::fmt::Debug {}

#[derive(Debug, Copy, Clone)]
pub enum ErrorKind {
    Msg(&'static str),
    Cancelled,
}

#[derive(Debug)]
pub struct LapJVError {
    kind: ErrorKind,
}

impl LapJVError {
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl std::fmt::Display for LapJVError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self.kind {
            ErrorKind::Msg(string) => write!(f, "{}", string),
            ErrorKind::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for LapJVError {}

#[derive(Default)]
pub struct LapJV<T> {
    free_rows: Vec<usize>,
    free_rows_util: Vec<usize>,
    v: Vec<T>,
    in_col: Vec<usize>,
    in_row: Vec<usize>,
    unique: Vec<bool>,
    in_row_not_set: Vec<bool>,
    pred: Vec<usize>,
    collist: Vec<usize>,
    cost_distance: Vec<T>,
}

/// Calculate solution cost by a result row
pub fn cost<T>(input: DMatrixView<T>, row: &[usize]) -> T
where
    T: LapJVCost,
{
    (0..row.len()).fold(T::zero(), |acc, i| acc + input[(i, row[i])])
}

/// Solve LAP problem given cost matrix
/// This is an implementation of the LAPJV algorithm described in:
/// R. Jonker, A. Volgenant. A Shortest Augmenting Path Algorithm for
/// Dense and Sparse Linear Assignment Problems. Computing 38, 325-340
/// (1987)
impl<'a, T> LapJV<T>
where
    T: LapJVCost + 'static,
{
    pub fn solve(
        &'a mut self,
        costs: DMatrixView<T>,
    ) -> Result<(&'a Vec<usize>, &'a Vec<usize>), LapJVError> {
        if !costs.is_square() {
            return Err(LapJVError {
                kind: ErrorKind::Msg("Input error: matrix is not square"),
            });
        }

        let dim = costs.shape().0; // square matrix dimensions

        self.free_rows.reserve(dim);
        self.free_rows.clear();

        self.v.clear();
        self.v.reserve(dim);

        self.in_row.clear();
        self.in_row.resize(dim, 0);

        self.in_col.clear();
        self.in_col.reserve(dim);

        self.ccrrt_dense(&costs);

        let mut i = 0;
        while !self.free_rows.is_empty() && i < 2 {
            self.carr_dense(&costs);
            i += 1;
        }

        if !self.free_rows.is_empty() {
            self.ca_dense(&costs)?;
        }

        Ok((&self.in_row, &self.in_col))
    }

    // Column-reduction and reduction transfer for a dense cost matrix
    fn ccrrt_dense(&'a mut self, costs: &DMatrixView<T>) {
        for col in costs.column_iter() {
            let (min_index, min_value) = col.iter().enumerate().skip(1).fold(
                (0, col[0]),
                |(old_idx, old_min), (new_idx, new_min)| {
                    if *new_min < old_min {
                        (new_idx, *new_min)
                    } else {
                        (old_idx, old_min)
                    }
                },
            );

            self.in_col.push(min_index);
            self.v.push(min_value);
        }

        self.unique.clear();
        self.unique.resize(costs.shape().0, true);

        self.in_row_not_set.clear();
        self.in_row_not_set.resize(costs.shape().0, true);

        for j in (0..costs.shape().0).rev() {
            let i = self.in_col[j];
            if self.in_row_not_set[i] {
                self.in_row[i] = j;
                self.in_row_not_set[i] = false;
            } else {
                self.unique[i] = false;
                self.in_col[j] = std::usize::MAX;
            }
        }

        for i in 0..costs.shape().0 {
            if self.in_row_not_set[i] {
                self.free_rows.push(i);
            } else if self.unique[i] {
                let j = self.in_row[i];
                let mut min = T::max_value();
                for j2 in 0..costs.shape().0 {
                    if j2 == j {
                        continue;
                    }
                    let c = reduced_cost(costs, &mut self.v, i, j2);
                    if c < min {
                        min = c;
                    }
                }

                self.v[j] -= min;
            }
        }
    }

    // Augmenting row reduction for a dense cost matrix
    fn carr_dense(&'a mut self, costs: &DMatrixView<T>) {
        // AUGMENTING ROW REDUCTION
        // scan all free rows.
        // in some cases, a free row may be replaced with another one to be scanned next.
        trace!("carr_dense");
        let dim = costs.shape().0;
        let mut current = 0;
        let mut new_free_rows = 0; // start list of rows still free after augmenting row reduction.
        let mut rr_cnt = 0;
        let num_free_rows = self.free_rows.len();

        while current < num_free_rows {
            rr_cnt += 1;
            let free_i = self.free_rows[current];
            current += 1;
            // find minimum and second minimum reduced cost over columns.
            let row = costs.row(free_i);
            let (v1, v2, mut j1, j2) = find_umins_plain(row.as_view(), &self.v);

            let mut i0 = self.in_col[j1];
            let v1_new = self.v[j1] - (v2 - v1);
            let v1_lowers = v1_new < self.v[j1]; // the trick to eliminate the epsilon bug

            if rr_cnt < current * dim {
                if v1_lowers {
                    // change the reduction of the minimum column to increase the minimum
                    // reduced cost in the row to the subminimum.
                    self.v[j1] = v1_new;
                } else if i0 != std::usize::MAX && j2.is_some() {
                    // minimum and subminimum equal.
                    // minimum column j1 is assigned.
                    // swap columns j1 and j2, as j2 may be unassigned.
                    j1 = j2.unwrap();
                    i0 = self.in_col[j1];
                }
                if i0 != std::usize::MAX {
                    // minimum column j1 assigned earlier.
                    if v1_lowers {
                        // put in current k, and go back to that k.
                        // continue augmenting path i - j1 with i0.
                        current -= 1;
                        self.free_rows[current] = i0;
                    } else {
                        // no further augmenting reduction possible.
                        // store i0 in list of free rows for next phase.
                        self.free_rows[new_free_rows] = i0;
                        new_free_rows += 1;
                    }
                }
            } else if i0 != std::usize::MAX {
                self.free_rows[new_free_rows] = i0;
                new_free_rows += 1;
            }
            self.in_row[free_i] = j1;
            self.in_col[j1] = free_i;
        }
        self.free_rows.truncate(new_free_rows);
    }

    // Augment for a dense cost matrix
    fn ca_dense(&'a mut self, costs: &DMatrixView<T>) -> Result<(), LapJVError> {
        let dim = costs.shape().0;
        self.pred.clear();
        self.pred.resize(dim, 0);

        self.free_rows_util.clear();
        self.free_rows_util.clone_from(&self.free_rows);
        self.free_rows.clear();

        for freerow in self.free_rows_util.iter() {
            trace!("looking at freerow={}", freerow);

            let mut i = std::usize::MAX;
            let mut k = 0;
            let mut j = find_path_dense(
                costs,
                *freerow,
                &mut self.collist,
                &mut self.cost_distance,
                &mut self.pred,
                &mut self.in_col,
                &mut self.v,
            );
            debug_assert!(j < dim);
            while i != *freerow {
                i = self.pred[j];
                self.in_col[j] = i;
                std::mem::swap(&mut j, &mut self.in_row[i]);
                k += 1;
                if k > dim {
                    return Err(LapJVError {
                        kind: ErrorKind::Msg("Error: ca_dense will not finish"),
                    });
                }
            }
        }
        Ok(())
    }
}

#[inline(always)]
fn reduced_cost<T: LapJVCost>(costs: &DMatrixView<T>, v: &mut Vec<T>, i: usize, j: usize) -> T {
    costs[(i, j)] - v[j]
}

// Scan all columns in TODO starting from arbitrary column in SCAN
// and try to decrease d of the TODO columns using the SCAN column
fn scan_dense<T: LapJVCost>(
    costs: &DMatrixView<T>,
    plo: &mut usize,
    phi: &mut usize,
    collist: &mut Vec<usize>,
    cost_distance: &mut Vec<T>,
    pred: &mut Vec<usize>,
    in_col: &mut Vec<usize>,
    v: &mut Vec<T>,
) -> Option<usize> {
    let mut lo = *plo;
    let mut hi = *phi;
    while lo != hi {
        let j = collist[lo];
        lo += 1;
        let i = in_col[j];
        let mind = cost_distance[j];
        let h = reduced_cost(costs, v, i, j) - mind;
        // For all columns in TODO
        for k in hi..collist.len() {
            let j = collist[k];
            let cred_ij = reduced_cost(costs, v, i, j) - h;
            if cred_ij < cost_distance[j] {
                cost_distance[j] = cred_ij;
                pred[j] = i;
                if (cred_ij - mind).abs() < T::epsilon() {
                    // if cred_ij == mind {
                    if in_col[j] == std::usize::MAX {
                        return Some(j);
                    }
                    collist[k] = collist[hi];
                    collist[hi] = j;
                    hi += 1;
                }
            }
        }
    }
    // Note: only change lo and hi if the item was not found
    *plo = lo;
    *phi = hi;

    None
}

/// Single iteration of modified Dijkstra shortest path algorithm as explained in the JV paper
/// return The closest free column index
fn find_path_dense<T: LapJVCost>(
    costs: &DMatrixView<T>,
    start_i: usize,
    collist: &mut Vec<usize>,
    cost_distance: &mut Vec<T>,
    pred: &mut Vec<usize>,
    in_col: &mut Vec<usize>,
    v: &mut Vec<T>,
) -> usize {
    let dim = costs.shape().0;

    collist.clear();
    collist.reserve(dim); // list of columns to be scanned in various ways.

    cost_distance.clear();
    cost_distance.reserve(dim); // 'cost-distance' in augmenting path calculation.

    let mut lo = 0;
    let mut hi = 0;
    let mut n_ready = 0;

    // Dijkstra shortest path algorithm.
    // runs until unassigned column added to shortest path tree.
    for i in 0..dim {
        collist.push(i);
        cost_distance.push(reduced_cost(costs, v, start_i, i));
        pred[i] = start_i;
    }

    trace!("d: {:?}", cost_distance);
    let mut final_j = None;
    while final_j.is_none() {
        if lo == hi {
            trace!("{}..{} -> find", lo, hi);
            n_ready = lo;
            hi = find_dense(dim, lo, &cost_distance, collist.as_mut_slice());
            trace!("check {}..{}", lo, hi);
            // check if any of the minimum columns happens to be unassigned.
            // if so, we have an augmenting path right away.
            for &j in collist.iter().take(hi).skip(lo) {
                if in_col[j] == std::usize::MAX {
                    final_j = Some(j);
                }
            }
        }

        if final_j.is_none() {
            trace!("{}..{} -> scan", lo, hi);
            final_j = scan_dense(
                costs,
                &mut lo,
                &mut hi,
                collist,
                cost_distance,
                pred,
                in_col,
                v,
            );
        }
    }

    trace!("found final_j={:?}", final_j);
    trace!("cols={:?}", collist);
    let mind = cost_distance[collist[lo]];
    for &j in collist.iter().take(n_ready) {
        v[j] += cost_distance[j] - mind;
    }
    final_j.unwrap()
}

fn find_dense<T>(dim: usize, lo: usize, d: &[T], collist: &mut [usize]) -> usize
where
    T: LapJVCost,
{
    let mut hi = lo + 1;
    let mut mind = d[collist[lo]];
    for k in hi..dim {
        let j = collist[k];
        let h = d[j];
        if h <= mind {
            if h < mind {
                // new minimum.
                hi = lo; // restart list at index low.
                mind = h;
            }
            // new index with same minimum, put on undex up, and extend list.
            collist[k] = collist[hi];
            collist[hi] = j;
            hi += 1;
        }
    }
    hi
}

// Finds minimum and second minimum from a row, returns (min, second_min, min_index, second_min_index)
#[inline(always)]
fn find_umins_plain<T>(local_cost: DMatrixView<T>, v: &[T]) -> (T, T, usize, Option<usize>)
where
    T: LapJVCost,
{
    let mut umin = local_cost[(0, 0)] - v[0];
    let mut usubmin = T::max_value();
    let mut j1 = 0;
    let mut j2 = None;
    for j in 1..local_cost.shape().1 {
        let h = local_cost[(0, j)] - v[j];
        if h < usubmin {
            if h >= umin {
                usubmin = h;
                j2 = Some(j);
            } else {
                usubmin = umin;
                umin = h;
                j2 = Some(j1);
                j1 = j;
            }
        }
    }
    (umin, usubmin, j1, j2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector, SMatrix};
    use rand::{self, Rng};

    #[test]
    fn it_works() {
        let mut solver = LapJV::default();

        // Solve once
        {
            let m: SMatrix<f32, 4, 4> = SMatrix::from_row_slice(&[
                1.0, 2.0, 3.0, 0.2, //
                4.0, 5.0, 6.0, 0.1, //
                7.0, 8.0, 9.0, 0.1, //
                0.2, 0.3, 0.4, 0.5, //
            ]);

            let result = solver.solve(m.as_view()).unwrap();
            assert_eq!(*result.0, vec![0, 1, 3, 2]);
            assert_eq!(*result.1, vec![0, 1, 3, 2]);
        }

        // Solve another problem, using the same solver.
        // This call should not allocate any memory, and instead use
        // the memory used in the previous execution.
        {
            let m: SMatrix<f32, 3, 3> = SMatrix::from_row_slice(&[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ]);

            let result = solver.solve(m.as_view()).unwrap();
            assert_eq!(*result.0, vec![2, 0, 1]);
            assert_eq!(*result.1, vec![1, 2, 0]);
        }
    }

    #[test]
    fn test_solve_random10() {
        let (m, result) = solve_random10();
        let cost = cost(m.as_view(), &result.0);
        assert_eq!(cost, 1071.0);
        assert_eq!(result.0, vec![7, 9, 3, 4, 1, 0, 5, 6, 2, 8]);
    }

    #[test]
    fn test_solve_inf1() {
        let c = &[
            std::f64::INFINITY,
            643.0,
            717.0,
            2.0,
            946.0,
            534.0,
            242.0,
            235.0,
            376.0,
            839.0,
            std::f64::INFINITY,
            141.0,
            799.0,
            180.0,
            386.0,
            745.0,
            592.0,
            822.0,
            421.0,
            42.0,
            std::f64::INFINITY,
            369.0,
            831.0,
            67.0,
            258.0,
            549.0,
            615.0,
            529.0,
            458.0,
            524.0,
            std::f64::INFINITY,
            649.0,
            287.0,
            910.0,
            12.0,
            820.0,
            31.0,
            92.0,
            217.0,
            555.0,
            std::f64::INFINITY,
            81.0,
            568.0,
            241.0,
            292.0,
            653.0,
            417.0,
            652.0,
            630.0,
            788.0,
            std::f64::INFINITY,
            822.0,
            788.0,
            166.0,
            122.0,
            690.0,
            304.0,
            568.0,
            449.0,
            214.0,
            std::f64::INFINITY,
            469.0,
            584.0,
            633.0,
            213.0,
            414.0,
            498.0,
            500.0,
            317.0,
            391.0,
            std::f64::INFINITY,
            581.0,
            183.0,
            420.0,
            16.0,
            748.0,
            35.0,
            516.0,
            639.0,
            356.0,
            std::f64::INFINITY,
            921.0,
            67.0,
            33.0,
            592.0,
            775.0,
            780.0,
            335.0,
            464.0,
            788.0,
            123.0,
            455.0,
            950.0,
            25.0,
            22.0,
            576.0,
            969.0,
            122.0,
            86.0,
            74.0,
        ];

        let m: SMatrix<f64, 10, 10> = SMatrix::from_row_slice(c);
        let mut solver = LapJV::default();
        let result = solver.solve(m.as_view()).unwrap();
        let cost = cost(m.as_view(), &result.0);

        assert_eq!(cost, 1403.0);
        assert_eq!(*result.0, vec![7, 9, 3, 8, 1, 4, 5, 6, 2, 0]);
    }

    #[test]
    fn test_find_umins() {
        let row = DVector::from_row_slice(&[25.0, 0.0, 15.0]).transpose();
        let result = find_umins_plain(row.as_view(), &vec![0.0, 0.0, 0.0]);
        assert_eq!(result, (0.0, 15.0, 1, Some(2)));
    }

    #[test]
    fn test_random() {
        let mut rng = rand::rng();
        let m: DMatrix<f64> = DMatrix::from_fn(10, 10, |_, _| rng.random::<f64>() * 100.0);
        let mut solver = LapJV::default();
        let _result = solver.solve(m.as_view()).unwrap();
    }

    fn solve_random10() -> (SMatrix<f64, 10, 10>, (Vec<usize>, Vec<usize>)) {
        let c = &[
            612.0, 643.0, 717.0, 2.0, 946.0, 534.0, 242.0, 235.0, 376.0, 839.0, //
            224.0, 141.0, 799.0, 180.0, 386.0, 745.0, 592.0, 822.0, 421.0, 42.0, //
            241.0, 369.0, 831.0, 67.0, 258.0, 549.0, 615.0, 529.0, 458.0, 524.0, //
            231.0, 649.0, 287.0, 910.0, 12.0, 820.0, 31.0, 92.0, 217.0, 555.0, //
            912.0, 81.0, 568.0, 241.0, 292.0, 653.0, 417.0, 652.0, 630.0, 788.0, //
            32.0, 822.0, 788.0, 166.0, 122.0, 690.0, 304.0, 568.0, 449.0, 214.0, //
            441.0, 469.0, 584.0, 633.0, 213.0, 414.0, 498.0, 500.0, 317.0, 391.0, //
            798.0, 581.0, 183.0, 420.0, 16.0, 748.0, 35.0, 516.0, 639.0, 356.0, //
            351.0, 921.0, 67.0, 33.0, 592.0, 775.0, 780.0, 335.0, 464.0, 788.0, //
            771.0, 455.0, 950.0, 25.0, 22.0, 576.0, 969.0, 122.0, 86.0, 74.0, //
        ];

        let m: SMatrix<f64, 10, 10> = SMatrix::from_row_slice(c);
        let mut solver = LapJV::default();
        let result = solver.solve(m.as_view()).unwrap();
        (m, (result.0.clone(), result.1.clone()))
    }

    #[test]
    fn dim_size_augmentation_path() {
        let m = &[
            849.096136535884,
            964.7344199800348,
            1658.3745235461179,
            1324.4750426251608,
            1565.0473271789378,
            1777.6465563492143,
            4280.139067225529,
            3411.9521087119633,
            1360.3260879628992,
            1546.701932942709,
            1304.724155636392,
            1048.3839719313205,
            1559.5777872153571,
            1769.1684309771547,
            3663.2542984837355,
            2926.089718214265,
        ];

        let m: SMatrix<f64, 4, 4> = SMatrix::from_row_slice(m);
        let mut solver = LapJV::default();
        let result = solver.solve(m.as_view());
        result.unwrap();
    }

    #[cfg(feature = "nightly")]
    mod benches {
        use super::*;
        use test::Bencher;

        #[bench]
        fn bench_solve_random10(b: &mut Bencher) {
            b.iter(|| test_solve_random10());
        }

        #[bench]
        fn bench_solve_random_inf1(b: &mut Bencher) {
            b.iter(|| test_solve_random10());
        }
    }
}
