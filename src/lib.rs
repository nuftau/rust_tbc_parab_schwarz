extern crate rand;
extern crate rayon;
#[macro_use]
extern crate ndarray;
extern crate ndarray_rand;
mod utils;
mod finite;

use rayon::prelude::*;
use ndarray::{Array, Array1, ArrayView1, ArrayView, stack, Axis};
use rand::distributions::uniform::Uniform;
use ndarray_rand::RandomExt;

pub unsafe fn construct_array_1d(input: *const f64, num_elems: usize) -> Array1<f64> {
    // Create a Rust `Vec` associated with a `numpy` array via the raw pointer. This is unsafe in Rust parlance
    // because Rust can't verify that the pointer actually points to legitimate memory, so to speak.
    // At least check that the pointer is not null.
    assert!(!input.is_null());
    ArrayView::from_shape_ptr(num_elems, input).to_owned()
}

#[allow(non_snake_case)]
/// Note: we won't be able to verify the size of h and D...
#[no_mangle]
pub extern fn rate(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : *const f64, h2 : *const f64,
                    D1 : *const f64, D2 : *const f64, is_finite_differences : i32,
                    number_samples : usize) -> f64 {

    let dis = match is_finite_differences {
        1 => Discretization::FiniteDifferences,
        0 => Discretization::FiniteVolumes,
        _ => Discretization::FiniteDifferences
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
        rate_rust(time_window_len, Lambda_1, Lambda_2, a, c, dt,
             M1, M2, h1a, h2a, D1a, D2a, dis, number_samples)
    }
}


#[allow(non_snake_case)]
/// Note: we won't be able to verify the size of h and D...
#[no_mangle]
pub extern fn full_interface_err(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : *const f64, h2 : *const f64,
                    D1 : *const f64, D2 : *const f64, is_finite_differences : i32,
                    number_samples : usize) -> *const f64 {

    let dis = match is_finite_differences {
        1 => Discretization::FiniteDifferences,
        0 => Discretization::FiniteVolumes,
        _ => Discretization::FiniteDifferences
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
        let ret = interface_errors_rust_nosum(time_window_len, Lambda_1, Lambda_2, a, c, dt,
                 M1, M2, h1a, h2a, D1a, D2a, dis, number_samples);
        let ret_ptr = ret.as_ptr();
        std::mem::forget(ret);
        ret_ptr
    }
}

#[allow(non_snake_case)]
/// Note: we won't be able to verify the size of h and D...
#[no_mangle]
pub extern fn interface_err(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : *const f64, h2 : *const f64,
                    D1 : *const f64, D2 : *const f64, is_finite_differences : i32,
                    number_samples : usize) -> *const f64 {

    let dis = match is_finite_differences {
        1 => Discretization::FiniteDifferences,
        0 => Discretization::FiniteVolumes,
        _ => Discretization::FiniteDifferences
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
        let ret = interface_errors_rust(time_window_len, Lambda_1, Lambda_2, a, c, dt,
                 M1, M2, h1a, h2a, D1a, D2a, dis, number_samples);
        let ret_ptr = ret.as_ptr();
        std::mem::forget(ret);
        ret_ptr
    }
}


pub enum Discretization {
    FiniteDifferences,
    FiniteVolumes
}

#[allow(non_snake_case)]
pub fn interface_errors_rust(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : Array1<f64>, h2 : Array1<f64>,
                    D1 : Array1<f64>, D2 : Array1<f64>, dis : Discretization,
                    number_samples : usize)
    -> Array1<f64> {

    let all_err : Vec<Array1<f64>> =
        (1..number_samples+1).collect::<Vec<usize>>().par_iter()
        .map(|seed| -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> {
            let errors = match dis {
                Discretization::FiniteVolumes => 
                    interface_errors::<finite::Volumes>(time_window_len, *seed as u64,
                        Lambda_1, Lambda_2, a, c, dt, M1, M2,
                        &h1.view(), &h2.view(), &D1.view(), &D2.view()),
                Discretization::FiniteDifferences =>
                    interface_errors::<finite::Differences>(time_window_len, *seed as u64,
                        Lambda_1, Lambda_2, a, c, dt, M1, M2,
                        &h1.view(), &h2.view(), &D1.view(), &D2.view())
            };
            stack(Axis(0), &[errors[0].view(), errors[1].view(), errors[2].view()]).unwrap()})
        .collect();
    let mut ret = Option::None;
    for err in all_err {
        if ret.is_some() {
            let mut value = ret.unwrap();
            value += &utils::linalg::abs(&err);
            ret = Option::Some(value);
        } else {
            ret = Some(utils::linalg::abs(&err));
        }
    }
    if ret.is_some() {
        let ret = ret.unwrap();
        ret / number_samples as f64
    } else {
        panic!("error: no samples generated.");
    }
}

#[allow(non_snake_case)]
pub fn interface_errors_rust_nosum(time_window_len : usize, Lambda_1 : f64,
                    Lambda_2 : f64, a : f64, c : f64, dt : f64,
                    M1 : usize, M2 : usize, h1 : Array1<f64>, h2 : Array1<f64>,
                    D1 : Array1<f64>, D2 : Array1<f64>, dis : Discretization,
                    number_samples : usize)
    -> Array1<f64> {

    let all_err : Vec<Array1<f64>> =
        (1..number_samples+1).collect::<Vec<usize>>().par_iter()
        .map(|seed| -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> {
            let errors = match dis {
                Discretization::FiniteVolumes => 
                    interface_errors::<finite::Volumes>(time_window_len, *seed as u64,
                        Lambda_1, Lambda_2, a, c, dt, M1, M2,
                        &h1.view(), &h2.view(), &D1.view(), &D2.view()),
                Discretization::FiniteDifferences =>
                    interface_errors::<finite::Differences>(time_window_len, *seed as u64,
                        Lambda_1, Lambda_2, a, c, dt, M1, M2,
                        &h1.view(), &h2.view(), &D1.view(), &D2.view())
            };
            stack(Axis(0), &[errors[0].view(), errors[1].view(), errors[2].view()]).unwrap()})
        .collect();
    let NUMBER_ERRORS = 3;
    let mut ret = Array::zeros(number_samples*NUMBER_ERRORS*time_window_len);
    let mut first_index = 0;
    for err in all_err {
        let next_index = first_index + NUMBER_ERRORS*time_window_len;
        ret.slice_mut(s![first_index..next_index]).assign(&err);
        first_index += NUMBER_ERRORS*time_window_len;
    }
    ret
}

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
fn interface_errors<T>(time_window_len : usize, _seed : u64, Lambda_1 : f64,
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
    let mut all_u1_interface = Array::random(time_window_len,
                                                Uniform::new_inclusive(-1.,1.));
    let mut all_phi1_interface = Array::random(time_window_len,
                                                Uniform::new_inclusive(-1.,1.));
    let mut all_u2_interface = Array1::zeros(time_window_len);
    let mut all_phi2_interface = Array1::zeros(time_window_len);

    all_u1_interface[time_window_len - 1] /= 1000.;
    all_phi1_interface[time_window_len - 1] /= 1000.;
    let mut ret = [all_u1_interface.to_owned(),
                    Array::linspace(0., 0., 0),
                    Array::linspace(0., 0., 0) ];

    for k in 1..3 {
        let mut all_u1_nm1 = Array::zeros(M1);
        let mut all_u2_nm1 = Array::zeros(M2);

        for i in 0..time_window_len {
            let u1_interface : f64 = all_u1_interface[i];
            let phi1_interface : f64 = all_phi1_interface[i];
            let ret_tuple_integrate =
                T::integrate_one_step(M2, h2, D2, a, c, dt, &f2,
                                      neumann, Lambda_2,
                                      &all_u2_nm1.view(),
                                      u1_interface, phi1_interface,
                                      &Y2, true);
            all_u2_nm1.assign(&ret_tuple_integrate.0);
            all_u2_interface[i] = ret_tuple_integrate.1;
            all_phi2_interface[i] = ret_tuple_integrate.2;
        }

        for i in 0..time_window_len {
            let u2_interface :f64 = all_u2_interface[i];
            let phi2_interface :f64 = all_phi2_interface[i];
            let ret_tuple_integrate =
                T::integrate_one_step(M1, h1, D1, a, c, dt, &f1,
                                      dirichlet, Lambda_1, &all_u1_nm1.view(),
                                      u2_interface, phi2_interface,
                                      &Y1, false);
            all_u1_nm1.assign(&ret_tuple_integrate.0);
            all_u1_interface[i] = ret_tuple_integrate.1;
            all_phi1_interface[i] = ret_tuple_integrate.2;
        }
        ret[k] = all_u1_interface.to_owned();
    }
    ret
}
