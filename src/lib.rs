extern crate lapack;
extern crate openblas_src;
extern crate ndarray;
use lapack::dgtsv;
use ndarray::Array1;
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        use crate::solve_linear;
        use ndarray::Array1;
        let y0 = 0.0 * Array1::ones(2);
        let y1 = 2.0 * Array1::ones(3);
        let y2 = 0.0 * Array1::ones(2);
        let f = 5.0 * Array1::ones(3);
        let x_star = 2.5 * Array1::ones(3);
        let x = solve_linear([&y0, &y1, &y2], &f);
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

pub fn solve_linear(matrix_to_invert : [&Array1<f64> ; 3], 
                f : &Array1<f64>) -> Array1<f64> {
    let y = matrix_to_invert;
    assert_eq!(y[0].len(), y[2].len());
    assert_eq!(f.len(), y[1].len());
    assert_eq!(y[1].len(), y[2].len()+1);
    let n = y[1].len() as i32;
    let nrhs = 1;
    let mut dl = y[0].clone();
    let mut d = y[1].clone();
    let mut du = y[2].clone();
    let mut x : Array1<f64> = f.clone(); // f input and also x output
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

