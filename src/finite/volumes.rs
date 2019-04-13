use crate::utils::linalg::{diff, solve_linear};
use ndarray::{stack, Array1, ArrayView1, Axis};

#[allow(dead_code)]
pub struct Volumes {}

#[allow(non_snake_case)]
#[allow(dead_code)]
fn integrate_one_step_star(
    M1: usize,
    M2: usize,
    h1: &ArrayView1<f64>,
    h2: &ArrayView1<f64>,
    D1: &ArrayView1<f64>,
    D2: &ArrayView1<f64>,
    a: f64,
    c: f64,
    dt: f64,
    f1: &ArrayView1<f64>,
    f2: &ArrayView1<f64>,
    neumann: f64,
    dirichlet: f64,
    u_nm1: &ArrayView1<f64>,
) -> (Array1<f64>, f64, f64) {
    assert!(dt > 0.);
    assert_eq!(D1.len(), M1 + 1);
    assert_eq!(D2.len(), M2 + 1);
    assert_eq!(h1.len(), M1);
    assert_eq!(h2.len(), M2);
    assert_eq!(f1.len(), M1);
    assert_eq!(f2.len(), M2);
    assert_eq!(u_nm1.len(), M1 + M2);
    let h1 = h1.slice(s![..;-1]);
    let f1 = f1.slice(s![..;-1]);
    let D1 = D1.slice(s![..;-1]);

    let f = stack![Axis(0), f1.view(), f2.view()];

    let Y = get_Y_star(M1, M2, &h1, &h2.view(), &D1, &D2.view(), a, c, dt);

    let rhs = dt / (1. + dt * c) * (diff(&f) + diff(u_nm1) / dt);
    let dirichlet = dirichlet - dt / (1. + dt * c) * (f[0] + u_nm1[0] / dt);
    let neumann = neumann * D2[D2.len() - 1];
    let rhs = stack![Axis(0), array![dirichlet], rhs.view(), array![neumann]];

    let Y_to_send = [&Y[0].view(), &Y[1].view(), &Y[2].view()];
    let phi_ret = solve_linear(Y_to_send, &rhs.view());
    let d1 = &phi_ret.slice(s![..M1 + 1]) / &D1; // we go until interface
    let d2 = &phi_ret.slice(s![M1..]) / D2; // we start from interface
    let d1_kp1 = d1.slice(s![1..]);
    let d2_kp1 = d2.slice(s![1..]);
    let d1_km1 = d1.slice(s![..d1.len() - 1]);
    let d2_km1 = d2.slice(s![..d2.len() - 1]);

    let D1_kp1 = D1.slice(s![1..]);
    let D2_kp1 = D2.slice(s![1..]);
    let D1_km1 = D1.slice(s![..D1.len() - 1]);
    let D2_km1 = D2.slice(s![..D2.len() - 1]);

    let u1_on_dt = &u_nm1.slice(s![..M1]) / dt;
    let u1_n = (&f.slice(s![..M1]) + &u1_on_dt + (&D1_kp1 * &d1_kp1 - &D1_km1 * &d1_km1) / &h1
        - a * (&d1_kp1 + &d1_km1) / 2.)
        * dt
        / (1. + dt * c);

    let u2_on_dt = &u_nm1.slice(s![M1..]) / dt;
    let u2_n = (&f.slice(s![M1..]) + &u2_on_dt + &(&D2_kp1 * &d2_kp1 - &D2_km1 * &d2_km1) / h2
        - a * (&d2_kp1 + &d2_km1) / 2.)
        * dt
        / (1. + dt * c);

    assert_eq!(u1_n.len(), M1);
    assert_eq!(u2_n.len(), M2);

    let u1_interface = u1_n[u1_n.len() - 1]
        + h1[h1.len() - 1] * d1[d1.len() - 2] / 6.
        + h1[h1.len() - 1] * d1[d1.len() - 1] / 3.;
    let u2_interface = u2_n[0] - h2[0] * d2[1] / 6. - h2[0] * d2[0] / 3.;
    // TODO remove one of the interfaces and the assert
    debug_assert!((u1_interface - u2_interface).abs() < 1e-5);
    let phi_interface = phi_ret[M1];
    let u_n = stack![Axis(0), u1_n.view(), u2_n.view()];

    (u_n, u1_interface, phi_interface)
}

#[allow(non_snake_case)]
#[allow(dead_code)]
pub fn get_Y(
    M: usize,
    h: &ArrayView1<f64>,
    D: &ArrayView1<f64>,
    a: f64,
    c: f64,
    dt: f64,
    Lambda: f64,
    upper_domain: bool,
) -> [Array1<f64>; 3] {
    assert_eq!(h.len(), M);
    assert_eq!(D.len(), M + 1);
    for &hi in h.iter() {
        assert!(hi > 0.);
    }
    for &Di in D.iter() {
        assert!(Di > 0.);
    }
    assert!(a >= 0.);
    assert!(c >= 0.);
    let Y;
    let mut Y_0;
    let mut Y_1;
    let mut Y_2;

    if upper_domain {
        Y = get_Y_star(
            1,
            M,
            &array![1.0].view(),
            h,
            &array![1., D[0]].view(),
            D,
            a,
            c,
            dt,
        );
        Y_0 = Y[0].slice(s![1..]).to_owned();
        Y_1 = Y[1].slice(s![1..]).to_owned();
        Y_2 = Y[2].slice(s![1..]).to_owned();

        // Now we have the tridiagonal matrices, except for the Robin bd condition
        let dirichlet_cond_extreme_point =
            -dt / (1. + dt * c) * (1. / h[0] + a / (2. * D[0])) - h[0] / (3. * D[0]);
        let dirichlet_cond_interior_point =
            dt / (1. + dt * c) * (1. / h[0] - a / (2. * D[1])) - h[0] / (6. * D[1]);
        // Robin bd condition are Lambda * Dirichlet + Neumann:
        // Except we work with fluxes:
        // Neumann condition is actually a Dirichlet bd condition
        // and Dirichlet is just a... pseudo-differential operator
        Y_1[0] = Lambda * dirichlet_cond_extreme_point + 1.;
        Y_2[0] = Lambda * dirichlet_cond_interior_point;
    } else {
        Y = get_Y_star(
            M,
            1,
            h,
            &array![1.].view(),
            D,
            &array![D[0], 1.].view(),
            a,
            c,
            dt,
        );
        // Here Y_0 and Y_2 are inverted because we need to take the symmetric
        Y_0 = Y[0].slice(s![..Y[0].len() - 1]).to_owned();
        Y_1 = Y[1].slice(s![..Y[1].len() - 1]).to_owned();
        Y_2 = Y[2].slice(s![..Y[2].len() - 1]).to_owned();
        // Now we have the tridiagonal matrices, except for the Robin bd condition
        let dirichlet_cond_extreme_point = dt / (1. + dt * c)
            * (1. / h[h.len() - 1] - a / (2. * D[D.len() - 1]))
            + h[h.len() - 1] / (3. * D[D.len() - 1]);
        let dirichlet_cond_interior_point = dt / (1. + dt * c)
            * (-1. / h[h.len() - 1] - a / (2. * D[D.len() - 2]))
            + h[h.len() - 1] / (6. * D[D.len() - 2]);
        // Robin bd condition are Lambda * Dirichlet + Neumann:
        // Except we work with fluxes:
        // Neumann condition is actually a Dirichlet bd condition
        // and Dirichlet is just a... pseudo-differential operator
        Y_1[M] = Lambda * dirichlet_cond_extreme_point + 1.;
        Y_0[M - 1] = Lambda * dirichlet_cond_interior_point;
        // We don't take the flipped, symmetric of the matrix.
        // this will be taken care of in integrate_...
    }

    assert_eq!(Y_1.len(), M + 1);
    assert_eq!(Y_2.len(), M);
    assert_eq!(Y_0.len(), M);
    let ret: [Array1<f64>; 3] = [Y_0, Y_1, Y_2];
    ret
}

/// D1[0] is the bottom of ocean. h1 is positive.
#[allow(non_snake_case)]
#[allow(dead_code)]
fn get_Y_star(
    M1: usize,
    M2: usize,
    h1: &ArrayView1<f64>,
    h2: &ArrayView1<f64>,
    D1: &ArrayView1<f64>,
    D2: &ArrayView1<f64>,
    a: f64,
    c: f64,
    dt: f64,
) -> [Array1<f64>; 3] {
    assert!(dt > 0.);
    assert_eq!(D1.len(), M1 + 1);
    assert_eq!(D2.len(), M2 + 1);
    assert_eq!(h1.len(), M1);
    assert_eq!(h2.len(), M2);

    for &hi in h1.iter() {
        assert!(hi > 0.);
    }
    for &Di in D1.iter() {
        assert!(Di > 0.);
    }
    for &hi in h2.iter() {
        assert!(hi > 0.);
    }
    for &Di in D2.iter() {
        assert!(Di > 0.);
    }
    assert!(a >= 0.);
    assert!(c >= 0.);

    let D_minus = stack![Axis(0), D1.slice(s![1..]), D2.slice(s![1..D2.len() - 1])];
    let D_plus = stack![
        Axis(0),
        D1.slice(s![1..D1.len() - 1]),
        D2.slice(s![..D2.len() - 1])
    ];
    let D_mm1_2 = stack![
        Axis(0),
        D1.slice(s![..D1.len() - 1]),
        D2.slice(s![..D2.len() - 2])
    ];
    let D_mp3_2 = stack![Axis(0), D1.slice(s![2..]), D2.slice(s![1..])];

    let h = stack![Axis(0), h1.view(), h2.view()];
    let h_m = h.slice(s![..h.len() - 1]);
    let h_mp1 = h.slice(s![1..]);

    /////// MAIN DIAGONAL
    let dirichlet =
        array![-dt / (1. + dt * c) * (1. / h[0] + a / (2. * D1[0])) - h[0] / (3. * D1[0])];
    let neumann = array![1.];
    let Y_1 = stack![
        Axis(0),
        dirichlet,
        dt / (1. + dt * c) * (1. / &h_m + 1. / &h_mp1 + a / 2. * (1. / &D_plus - 1. / &D_minus))
            + (&h_m / &D_minus + &h_mp1 / &D_plus) / 3.,
        neumann
    ];

    //////// RIGHT DIAGONAL
    let dirichlet =
        array![dt / (1. + dt * c) * (1. / h[0] - a / (2. * D1[1])) - h[0] / (6. * D1[1])];
    let Y_2 = stack![
        Axis(0),
        dirichlet.view(),
        (-dt / (1. + dt * c) * (1. / &h_mp1 - a / (2. * &D_mp3_2)) + 1. / (6. * &D_mp3_2) * &h_mp1)
            .view()
    ];

    //////// LEFT DIAGONAL
    let neumann = array![0.]; // Neumann bd condition (achtually dirichlet)
    let Y_0 = stack![
        Axis(0),
        -dt / (1. + dt * c) * (1. / &h_m + a / (2. * &D_mm1_2)) + 1. / (6. * &D_mm1_2) * &h_m,
        neumann
    ];

    assert_eq!(Y_1.len(), M1 + M2 + 1);
    assert_eq!(Y_2.len(), M1 + M2);
    assert_eq!(Y_0.len(), M1 + M2);
    let ret: [Array1<f64>; 3] = [Y_0, Y_1, Y_2];
    ret
}
