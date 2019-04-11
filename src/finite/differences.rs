use ndarray::{Array, Array1, Axis, stack, ArrayView1};
use crate::utils::linalg::solve_linear;

#[allow(non_snake_case)]
#[allow(dead_code)]
pub fn integrate_one_step_star(M1 : usize, M2 : usize,
                               h1 : &ArrayView1<f64>,
                               h2 : &ArrayView1<f64>,
                               D1 : &ArrayView1<f64>,
                               D2 : &ArrayView1<f64>,
                               a : f64, c : f64,
                               dt : f64,
                               f1 : &ArrayView1<f64>,
                               f2 : &ArrayView1<f64>,
                               neumann : f64,
                               dirichlet : f64,
                               u_nm1 : &ArrayView1<f64>) ->
                         (Array1<f64>, f64, f64) {
    assert!(dt > 0.);
    assert_eq!(D1.len(), M1-1);
    assert_eq!(D2.len(), M2-1);
    assert_eq!(h1.len(), M1-1);
    assert_eq!(h2.len(), M2-1);
    assert_eq!(f1.len(), M1);
    assert_eq!(f2.len(), M2);
    assert_eq!(u_nm1.len(), M1 + M2 - 1);
    let h1f = h1.slice(s![..;-1]);
    let f1f = f1.slice(s![..;-1]);
    let D1f = D1.slice(s![..;-1]);
    let middle_f = array![f1f[f1f.len()-1] * (-h1f[h1f.len()-1])
        / (-h1f[h1f.len()-1] + h2[0]) 
        + f2[0] * h2[0] / (-h1f[h1f.len()-1] + h2[0])];
    let D = stack(Axis(0), &[D1f.view(), D2.view()]).expect("Cannot stack D1 and D2");
    let h = stack(Axis(0), &[(Array::zeros(h1f.raw_dim())-h1f).view(),
                                            h2.view()]).expect("Cannot stack h1 and h2");
    let f = stack![Axis(0), f1f.slice(s![..f1f.len()-1]),
            middle_f,
            f2.slice(s![1..])];
    let M = M1 + M2 - 1;

    let sum_both_h = &h.slice(s![1..])
            + &h.slice(s![..h.len()-1]);
    let mut Y = get_Y_star(M, &h.view(), &D.view(), a, c);
    let mut Y_1_interior = Y[1].slice_mut(s![1..Y[1].len()-1]);
    Y_1_interior += &(&sum_both_h / dt);

    let Y_0 = &Y[0]; let Y_2 = &Y[2];

    let rhs = stack![Axis(0), array![dirichlet],
        sum_both_h * (&f.slice(s![1..f.len()-1])
               + &(&u_nm1.slice(s![1..u_nm1.len()-1]) / dt)),
               array![neumann]];

    let u_n = solve_linear([&Y_0.view(), &Y[1].view(), &Y_2.view()], &rhs.view());

    let u1_n = u_n.slice(s![..M1]);
    let u1_n = u1_n.slice(s![..;-1]);
    let u2_n = u_n.slice(s![M1-1..]);

    assert_eq!(u2_n.len(), M2);

    let phi_interface1 = D1[0]/h1[0] * (u1_n[1] - u1_n[0])
        - h1[0] / 2. * ((u1_n[0]-u_nm1[M1-1])/dt + a*(u1_n[1])/h1[0]
                      + c * u1_n[0] - f1[0]);
    // phi_interface2 has the same value: I put it here for reference:
    //let phi_interface2 = D2[0]/h2[0] * (u2_n[1] - u2_n[0])
    //    - h2[0] / 2. * ((u2_n[0]-u_nm1[M1-1])/dt + a*(u2_n[1])/h2[0]
    //                  + c * u2_n[0] - f2[0]);
    let u_interface = u_n[M1-1];

    (u_n, u_interface, phi_interface1)
}


#[allow(non_snake_case)]
#[allow(dead_code)]
pub fn get_Y(M : usize, Lambda : f64, h : &ArrayView1<f64>,
                  D : &ArrayView1<f64>, a : f64,
                  c : f64, dt : f64, upper_domain : bool)
        ->  [Array1<f64> ; 3] {
    assert_eq!(h.len(), M - 1);
    assert_eq!(D.len(), M - 1);
    if upper_domain { for &hi in h.iter() { assert!(hi > 0.); } }
    else { for &hi in h.iter() { assert!(hi < 0.); } }
    for &Di in D.iter() { assert!(Di > 0.); }
    assert!(a >= 0.);
    assert!(c >= 0.);

    let h_m = h.slice(s![1..]);
    let h_mm1 = h.slice(s![..h.len()-1]);
    let sum_both_h = &h_m + &h_mm1;
    let D_mp1_2 = D.slice(s![1..]);
    let D_mm1_2 = D.slice(s![..D.len()-1]);

    /////// MAIN DIAGONAL
    let bd_cond = match upper_domain {
        true => 1. / h[h.len()-1],
        false => 1. };
    let boundary = array![bd_cond];

    let corrective_term = h[0] / 2. * (1. / dt + c) - a / 2.;
    let interface = array![Lambda - D[0] / h[0] - corrective_term];

    let Y_1 = stack![Axis(0), interface,
        &sum_both_h*c +
            (&h_mm1*&D_mp1_2 + &h_m* &D_mm1_2) * 2.
                / (&h_m*&h_mm1),
        boundary ];

    //////// RIGHT DIAGONAL
    let interface = array![D[0] / h[0] - a / 2.];
    let Y_2 = stack(Axis(0), &[interface.view(),
        (&D_mp1_2*(-2.) / &h_m + a).view()]).expect("Cannot stack Y_2.");

    //////// LEFT DIAGONAL
    let bd_cond = match upper_domain {
        false => 0.,
        true => -1./h[h.len()-1]
    };
    let boundary = array![bd_cond];
    let Y_0 = stack![Axis(0),
        &D_mm1_2 * (-2.) / &h_mm1 - a, boundary];

    assert_eq!(Y_1.len(), M);
    assert_eq!(Y_2.len(), M - 1);
    assert_eq!(Y_0.len(), M - 1);
    let ret : [Array1<f64>;3] = [Y_0, Y_1, Y_2];
    ret 
}


#[allow(non_snake_case)]
#[allow(dead_code)]
pub fn get_Y_star(M_star : usize, h_star : &ArrayView1<f64>,
                  D_star : &ArrayView1<f64>, a : f64,
                  c : f64) ->  [Array1<f64> ; 3] {
    let M = M_star;
    let h = h_star;
    let D = D_star;
    assert_eq!(h.len(), M_star - 1);
    assert_eq!(D.len(), M_star - 1);
    for &hi in h.iter() { assert!(hi > 0.); }
    for &Di in D.iter() { assert!(Di > 0.); }
    assert!(a >= 0.);
    assert!(c >= 0.);

    let h_m = h.slice(s![1..]);
    let h_mm1 = h.slice(s![..h.len()-1]);
    let sum_both_h = &h_m + &h_mm1;
    let D_mp1_2 = D.slice(s![1..]);
    let D_mm1_2 = D.slice(s![..D.len()-1]);

    /////// MAIN DIAGONAL
    let dirichlet = array![1.];
    let neumann = array![1./h[h.len()-1]];
    let Y_1 = stack![Axis(0), dirichlet,
        &sum_both_h*c +
            (&h_mm1*&D_mp1_2 + &h_m* &D_mm1_2) * 2.
                / (&h_m*&h_mm1),
        neumann ];

    //////// RIGHT DIAGONAL
    let dirichlet = array![0.];
    let Y_2 = stack(Axis(0), &[dirichlet.view(),
        (&D_mp1_2*(-2.) / &h_m + a).view()]).expect("Cannot stack Y_2.");

    //////// LEFT DIAGONAL
    let neumann = array![-1./h[h.len()-1]];
    let Y_0 = stack![Axis(0),
        &D_mm1_2 * (-2.) / &h_mm1 - a, neumann];
    assert_eq!(Y_1.len(), M);
    assert_eq!(Y_2.len(), M - 1);
    assert_eq!(Y_0.len(), M - 1);
    let ret : [Array1<f64>;3] = [Y_0, Y_1, Y_2];
    ret 
}
