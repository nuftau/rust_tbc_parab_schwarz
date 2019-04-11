extern crate lapack;
extern crate openblas_src;
extern crate ndarray;
use lapack::dgtsv;
use ndarray::{ArrayView1, ArrayBase, Array1, Array, Data, Slice, Axis};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        use super::solve_linear;
        use ndarray::Array1;
        let y0 :Array1<f64> = 0.0 * Array1::ones(2);
        let y1 :Array1<f64> = 2.0 * Array1::ones(3);
        let y2 :Array1<f64> = 0.0 * Array1::ones(2);
        let f :Array1<f64> = 5.0 * Array1::ones(3);
        let x_star :Array1<f64> = 2.5 * Array1::ones(3);
        let x = solve_linear([&y0.view(), &y1.view(), &y2.view()], &f.view());
        assert!( x.all_close(&x_star, 0.001));
    }
}

///  Solve the linear system Yu = f and returns u.
///  This function is just a wrapper over scipy
///  matrix_to_invert is a tuple (Y_0, Y_1, Y_2) containing respectively:
///  - The left diagonal of size M-1
///  - The main diagonal of size M
///  - The right diagonal of size M-1
///
///  f is an array of size M.
///  f[0] is the condition on the bottom of the domain
///  f[-1] is the condition on top of the domain
///
///  /!\ f[1:-1] should be equal to f * (hm + hmm1) /!\
///  matrix_to_invert is returned by the functions get_Y and get_Y_star
/*
pub fn solve_linear(matrix_to_invert : [&ArrayView1<f64> ; 3], 
                f : &ArrayView1<f64>) -> Array1<f64> {
    use rgsl::linear_algebra::solve_tridiag;
    use rgsl::types::vector::VectorF64;

    let y = matrix_to_invert;
    assert_eq!(y[0].len(), y[2].len());
    assert_eq!(f.len(), y[1].len());
    assert_eq!(y[1].len(), y[2].len()+1);
    let y_0 = VectorF64::from_slice(y[0].into_slice());
    let y_1 = VectorF64::from_slice(y[1].into_slice());
    let y_2 = VectorF64::from_slice(y[2].into_slice());
    let b = VectorF64::from_slice(f.into_slice());
    let x = VectorF64::new(y_1.len());
    solve_tridiag(&y_1, &y_2, &y_0, &b, &x);
    x
}
*/
pub fn solve_linear(matrix_to_invert : [&ArrayView1<f64> ; 3], 
                f : &ArrayView1<f64>) -> Array1<f64> {
    let y = matrix_to_invert;
    assert_eq!(y[0].len(), y[2].len());
    assert_eq!(f.len(), y[1].len());
    assert_eq!(y[1].len(), y[2].len()+1);
    let n = y[1].len() as i32;
    let nrhs = 1;
    let mut dl = y[0].to_owned();
    let mut d = y[1].to_owned();
    let mut du = y[2].to_owned();
    let mut x : Array1<f64> = f.to_owned(); // f input and also x output
    let ldb = n;
    let mut info = 0;
    unsafe {
        dgtsv(n, nrhs,
              &mut dl.as_slice_mut().expect("dl cannot be taken as slice"),
              &mut d.as_slice_mut().expect("d cannot be taken as slice"),
              &mut du.as_slice_mut().expect("du cannot be taken as slice"),
              &mut x.as_slice_mut().expect("x cannot be taken as slice"),
              ldb, &mut info);
    }
    assert_eq!(info, 0);
    x
}

/// Same as np.diff(x). return x[k+1] - x[k].
pub fn diff<S, D>(x : &ArrayBase<S, D>) -> Array<f64, D> 
    where S : Data<Elem=f64>,  D : ndarray::Dimension, {
    x.slice_axis(Axis(0), Slice::from(1..)).to_owned() 
        - x.slice_axis(Axis(0), Slice::from(..x.len()-1))
}

/// Same as np.cos(x)
#[allow(dead_code)]
pub fn cos<S, D>(x : &ArrayBase<S, D>) -> Array<f64, D> 
    where S : Data<Elem=f64>, D : ndarray::Dimension {
    x.to_owned().map(|a| f64::cos(*a))
}

/// Same as np.cos(x)
#[allow(dead_code)]
pub fn abs<S, D>(x : &ArrayBase<S, D>) -> Array<f64, D> 
    where S : Data<Elem=f64>, D : ndarray::Dimension {
    x.to_owned().map(|a| f64::abs(*a))
}

/// Same as np.sin(x)
#[allow(dead_code)]
pub fn sin<S, D>(x : &ArrayBase<S, D>) -> Array<f64, D> 
    where S : Data<Elem=f64>, D : ndarray::Dimension {
    x.to_owned().map(|a| f64::sin(*a))
}

/// Same as np.linalg.norm(x). return x[k+1] - x[k].
pub fn norm<S, D>(x : &ArrayBase<S, D>) -> f64
    where S : Data<Elem=f64>, D : ndarray::Dimension {
    f64::sqrt((x*x).sum())
}

/// flips the slice
pub fn flip_if<S>(condition : bool, x : ArrayView1<S>)
    -> ArrayView1<S> {
    match condition {
        true => x.slice_move(s![..;-1]),
        false => x
    }
}

