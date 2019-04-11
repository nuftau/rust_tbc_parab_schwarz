extern crate rand;
extern crate rand_xorshift;
extern crate rayon;
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
mod utils;
pub mod finite;
#[macro_use]
extern crate cpython;

use cpython::{Python, PyResult};

use rayon::prelude::*;
use rand::SeedableRng;
use ndarray::{Array, Array1, ArrayView1};
use ndarray_rand::RandomExt;
use rand::distributions::uniform::Uniform;

#[cfg(test)]
mod tests {
    use crate::{Discretization, rate_rust_cst};
    #[test]
    fn test_rate() {
        println!("différences finies: {}", rate_rust_cst(150, 3., 0., 0., 1e-10, 0.1, 400, 400,
             -0.50125313, 0.50125313, 0.6, 0.54, Discretization::FiniteDifferences,
             100));
        println!("volumes finis: {}", rate_rust_cst(1, 3., 0., 0., 1e-10, 0.1, 400, 400,
             0.5, 0.5, 0.6, 0.54, Discretization::FiniteVolumes, 1));
        assert_eq!(3, 2+2 );
    }
}

pub unsafe fn construct_array_1d(input: *mut f64, num_elems: usize) -> Array1<f64> {
    // Create a Rust `Vec` associated with a `numpy` array via the raw pointer. This is unsafe in Rust parlance
    // because Rust can't verify that the pointer actually points to legitimate memory, so to speak.
    // At least check that the pointer is not null.
    assert!(!input.is_null());
    Array1::from_vec(Vec::from_raw_parts(input, num_elems, num_elems))
}

#[allow(non_snake_case)]
/// Note: we won't be able to verify the size of h and D...
fn rate(_py: Python, time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : *mut f64, h2 : *mut f64,
                    D1 : *mut f64, D2 : *mut f64, is_finite_differences : bool,
                    number_samples : usize) -> PyResult<f64> {

    let dis = match is_finite_differences {
        true => Discretization::FiniteDifferences,
        false => Discretization::FiniteVolumes
    };
    let sizes = match dis {
        Discretization::FiniteDifferences =>(M1-1, M2-1, M1-1, M2-1),
        Discretization::FiniteVolumes => (M1, M2, M1+1, M2+1),
    };

    unsafe {
        let h1a = construct_array_1d(h1, sizes.0);
        let h2a = construct_array_1d(h2, sizes.1);
        let D1a = construct_array_1d(D1, sizes.2);
        let D2a = construct_array_1d(D2, sizes.3);

        Ok(rate_rust(time_window_len, Lambda_1, Lambda_2, a, c, dt,
             M1, M2, h1a, h2a, D1a, D2a, 
             match is_finite_differences {
                 true => Discretization::FiniteDifferences,
                 false => Discretization::FiniteVolumes},
                 number_samples))
    }
}

pub enum Discretization {
    FiniteDifferences,
    FiniteVolumes
}

#[allow(non_snake_case)]
fn rate_cst(_py: Python, time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : f64, h2 : f64,
                    D1 : f64, D2 : f64, is_finite_differences : bool,
                    number_samples : usize) -> PyResult<f64> {
    Ok(rate_rust_cst(time_window_len, Lambda_1, Lambda_2, a, c, dt,
         M1, M2, h1, h2, D1, D2, 
         match is_finite_differences {
             true => Discretization::FiniteDifferences,
             false => Discretization::FiniteVolumes},
             number_samples))
}

// add bindings to the generated python module
// N.B: names: "librust2py" must be the name of the `.so` or `.pyd` file
py_module_initializer!(librust_rate_constant, initlibrust_rate_constant,
                       PyInit_librust_rate_constant, |py, m| {
    m.add(py, "__doc__", "This module is the function rate implemented in Rust.")?;
    m.add(py, "rate_cst", py_fn!(py, rate_cst(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : f64, h2 : f64,
                    D1 : f64, D2 : f64, is_finite_differences : bool,
                    number_samples : usize)))?;
    m.add(py, "rate", py_fn!(py, rate_cst(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : *mut f64, h2 : *mut f64,
                    D1 : *mut f64, D2 : *mut f64, is_finite_differences : bool,
                    number_samples : usize)))?;
    Ok(())
});


#[allow(non_snake_case)]
pub fn rate_rust(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : Array1<f64>, h2 : Array1<f64>,
                    D1 : Array1<f64>, D2 : Array1<f64>, dis : Discretization,
                    number_samples : usize) -> f64 {
    use crate::utils::linalg::norm;

    let ret : f64 = match dis {
        Discretization::FiniteVolumes =>
    (1..number_samples+1).collect::<Vec<usize>>().par_iter()
        .map(|seed| {
            let errors = interface_errors::<finite::Volumes>(time_window_len, *seed as u64,
                        Lambda_1, Lambda_2, a, c, dt, M1, M2, &h1.view(), &h2.view(), &D1.view(), &D2.view());
            norm(&errors[2])/norm(&errors[1])})
        .sum(),
        Discretization::FiniteDifferences =>
    (1..number_samples+1).collect::<Vec<usize>>().par_iter()
        .map(|seed| {
            let errors = interface_errors::<finite::Differences>(time_window_len, *seed as u64,
                        Lambda_1, Lambda_2, a, c, dt, M1, M2, &h1.view(), &h2.view(), &D1.view(), &D2.view());
            norm(&errors[2])/norm(&errors[1])})
        .sum()
    };
    ret / number_samples as f64
}

#[allow(non_snake_case)]
pub fn rate_rust_cst(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : f64, h2 : f64,
                    D1 : f64, D2 : f64, dis : Discretization,
                    number_samples : usize) -> f64 {
    let sizes = match dis {
        Discretization::FiniteDifferences =>(M1-1, M2-1, M1-1, M2-1),
        Discretization::FiniteVolumes => (M1, M2, M1+1, M2+1),
    };
    let h1 = Array::from_elem(sizes.0, h1);
    let h2 = Array::from_elem(sizes.1, h2);
    let D1 = Array::from_elem(sizes.2, D1);
    let D2 = Array::from_elem(sizes.3, D2);
    rate_rust(time_window_len, Lambda_1, Lambda_2, a, c, dt, M1, M2, h1, 
              h2, D1, D2, dis, number_samples)
}


#[allow(non_snake_case)]
fn interface_errors<T>(time_window_len : usize, seed : u64, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize,
                    h1 : &ArrayView1<f64>, h2 : &ArrayView1<f64>,
                    D1 : &ArrayView1<f64>, D2 : &ArrayView1<f64>)
    -> [Array1<f64>; 3]
    where T : finite::Discretization{
    let f1 = Array1::zeros(M1);
    let f1 = f1.view();
    let f2 = Array1::zeros(M2);
    let f2 = f2.view();
    let neumann = 0.;
    let dirichlet = 0.;
    let Y1 = T::precompute_Y(M1, h1, D1, a, c, dt, &f1,
                            dirichlet, Lambda_1, false);
    let Y2 = T::precompute_Y(M2, h2, D2, a, c, dt, &f2,
                            neumann, Lambda_2, true);
    let mut all_u1_nm1 = Array::zeros((time_window_len+1, M1));
    let mut all_u2_nm1 = Array::zeros((time_window_len+1, M2));

    let seed = to_2bytes(seed);

    let mut rng = rand_xorshift::XorShiftRng::from_seed(seed);
    let mut all_u1_interface = Array::random_using(time_window_len,
                                                Uniform::new(-1.,1.), &mut rng);
    let mut all_phi1_interface = Array::random_using(time_window_len,
                                                Uniform::new(-1.,1.), &mut rng);
    all_u1_interface[0] = 1.;
    all_phi1_interface[0] = 1.;
    let mut all_u2_interface = Array1::zeros(time_window_len);
    let mut all_phi2_interface = Array1::zeros(time_window_len);

    let mut ret = [all_u1_interface.view().to_owned(),
                    all_u1_interface.view().to_owned(),
                    all_u1_interface.view().to_owned()];

    for k in 1..3 {
        for i in 0..time_window_len {
            let u1_interface : f64 = all_u1_interface[i];
            let phi1_interface : f64 = all_phi1_interface[i];
            let ret_tuple_integrate =
                T::integrate_one_step(M2, h2, D2, a, c, dt, &f2,
                                      neumann, Lambda_2,
                                      &all_u2_nm1.slice(s![i, ..]),
                                      u1_interface, phi1_interface,
                                      &Y2, true);
            all_u2_nm1.slice_mut(s![i+1, ..]).assign(&ret_tuple_integrate.0);
            all_u2_interface[i] = ret_tuple_integrate.1;
            all_phi2_interface[i] = ret_tuple_integrate.2;
        }

        for i in 0..time_window_len {
            let u2_interface :f64 = all_u2_interface[i];
            let phi2_interface :f64 = all_phi2_interface[i];
            let ret_tuple_integrate =
                T::integrate_one_step(M1, h1, D1, a, c, dt, &f1,
                                      neumann, Lambda_1, &all_u1_nm1.slice(s![i,..]),
                                      u2_interface, phi2_interface,
                                      &Y1, false);
            all_u1_nm1.slice_mut(s![i+1, ..]).assign(&ret_tuple_integrate.0);
            all_u1_interface[i] = ret_tuple_integrate.1;
            all_phi1_interface[i] = ret_tuple_integrate.2;
        }
        ret[k] = all_u1_interface.view().to_owned();
    }
    ret
}
fn to_2bytes(seed : u64) -> [u8;16]{
    [((seed << 1) % 256)as u8,
     ((seed << 8) % 256)as u8,
     ((seed << 16) % 256)as u8,
     ((seed << 24) % 256)as u8,
     ((seed << 32) % 256)as u8,
     ((seed << 40) % 256)as u8,
     ((seed << 48) % 256)as u8,
     ((seed << 56) % 256)as u8,
    ((1+seed << 1) % 256)as u8,
     ((1+seed << 8) % 256)as u8,
     ((1+seed << 16) % 256)as u8,
     ((1+seed << 24) % 256)as u8,
     ((1+seed << 32) % 256)as u8,
     ((1+seed << 40) % 256)as u8,
     ((1+seed << 48) % 256)as u8,
     ((1+seed << 56) % 256)as u8]
}
