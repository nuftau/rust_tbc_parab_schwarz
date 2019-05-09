mod differences;
mod differences_naive;
mod volumes;
use crate::utils::linalg::{diff, flip_if, solve_linear};
use ndarray::{stack, Array1, ArrayView1, Axis};

pub struct Differences {}
pub struct Differences_naive {}
pub struct Volumes {}

pub trait Discretization {
    #[allow(non_snake_case)]
    fn integrate_one_step(
        M: usize,
        h: &ArrayView1<f64>,
        D: &ArrayView1<f64>,
        a: f64,
        c: f64,
        dt: f64,
        f: &ArrayView1<f64>,
        bd_cond: f64,
        Lambda: f64,
        u_nm1: &ArrayView1<f64>,
        u_interface: f64,
        phi_interface: f64,
        Y: &[Array1<f64>; 3],
        upper_domain: bool,
    ) -> (Array1<f64>, f64, f64);
    #[allow(non_snake_case)]
    fn precompute_Y(
        M: usize,
        h: &ArrayView1<f64>,
        D: &ArrayView1<f64>,
        a: f64,
        c: f64,
        dt: f64,
        f: &ArrayView1<f64>,
        bd_cond: f64,
        Lambda: f64,
        upper_domain: bool,
    ) -> [Array1<f64>; 3];
}

impl Discretization for Differences_naive {
    #[allow(non_snake_case)]
    fn precompute_Y(
        M: usize,
        h: &ArrayView1<f64>,
        D: &ArrayView1<f64>,
        a: f64,
        c: f64,
        dt: f64,
        _f: &ArrayView1<f64>,
        _bd_cond: f64,
        Lambda: f64,
        upper_domain: bool,
    ) -> [Array1<f64>; 3] {
        let sum_both_h = &h.slice(s![1..]) + &h.slice(s![..h.len() - 1]);
        let mut Y = differences_naive::get_Y(M, Lambda, &h, &D, a, c, dt, upper_domain);
        let mut Y_1_interior = Y[1].slice_mut(s![1..Y[1].len() - 1]);
        Y_1_interior += &(&sum_both_h / dt);
        Y
    }

    #[allow(non_snake_case)]
    fn integrate_one_step(
        M: usize,
        h: &ArrayView1<f64>,
        D: &ArrayView1<f64>,
        a: f64,
        c: f64,
        dt: f64,
        f: &ArrayView1<f64>,
        bd_cond: f64,
        Lambda: f64,
        u_nm1: &ArrayView1<f64>,
        u_interface: f64,
        phi_interface: f64,
        Y: &[Array1<f64>; 3],
        _upper_domain: bool,
    ) -> (Array1<f64>, f64, f64) {
        assert!(dt > 0.);
        assert_eq!(D.len(), M - 1);
        assert_eq!(h.len(), M - 1);
        assert_eq!(f.len(), M);
        assert_eq!(u_nm1.len(), M);

        let sum_both_h = &h.slice(s![1..]) + &h.slice(s![..h.len() - 1]);

        let cond_robin = Lambda * u_interface + phi_interface;

        let rhs = stack![
            Axis(0),
            array![cond_robin],
            sum_both_h
                * (&f.slice(s![1..f.len() - 1]) + &(&u_nm1.slice(s![1..u_nm1.len() - 1]) / dt)),
            array![bd_cond]
        ];

        let u_n = solve_linear([&Y[0].view(), &Y[1].view(), &Y[2].view()], &rhs.view());

        assert_eq!(u_n.len(), M);
        let u_interface = u_n[0];

        let phi_interface = D[0] / h[0] * (u_n[1] - u_n[0]);

        (u_n, u_interface, phi_interface)
    }
}

impl Discretization for Differences {
    #[allow(non_snake_case)]
    fn precompute_Y(
        M: usize,
        h: &ArrayView1<f64>,
        D: &ArrayView1<f64>,
        a: f64,
        c: f64,
        dt: f64,
        _f: &ArrayView1<f64>,
        _bd_cond: f64,
        Lambda: f64,
        upper_domain: bool,
    ) -> [Array1<f64>; 3] {
        let sum_both_h = &h.slice(s![1..]) + &h.slice(s![..h.len() - 1]);
        let mut Y = differences::get_Y(M, Lambda, &h, &D, a, c, dt, upper_domain);
        let mut Y_1_interior = Y[1].slice_mut(s![1..Y[1].len() - 1]);
        Y_1_interior += &(&sum_both_h / dt);
        Y
    }

    #[allow(non_snake_case)]
    fn integrate_one_step(
        M: usize,
        h: &ArrayView1<f64>,
        D: &ArrayView1<f64>,
        a: f64,
        c: f64,
        dt: f64,
        f: &ArrayView1<f64>,
        bd_cond: f64,
        Lambda: f64,
        u_nm1: &ArrayView1<f64>,
        u_interface: f64,
        phi_interface: f64,
        Y: &[Array1<f64>; 3],
        _upper_domain: bool,
    ) -> (Array1<f64>, f64, f64) {
        assert!(dt > 0.);
        assert_eq!(D.len(), M - 1);
        assert_eq!(h.len(), M - 1);
        assert_eq!(f.len(), M);
        assert_eq!(u_nm1.len(), M);

        let sum_both_h = &h.slice(s![1..]) + &h.slice(s![..h.len() - 1]);

        let cond_robin = Lambda * u_interface + phi_interface - h[0] / 2. * (u_nm1[0] / dt + f[0]);

        let rhs = stack![
            Axis(0),
            array![cond_robin],
            sum_both_h
                * (&f.slice(s![1..f.len() - 1]) + &(&u_nm1.slice(s![1..u_nm1.len() - 1]) / dt)),
            array![bd_cond]
        ];

        let u_n = solve_linear([&Y[0].view(), &Y[1].view(), &Y[2].view()], &rhs.view());

        assert_eq!(u_n.len(), M);
        let u_interface = u_n[0];

        let phi_interface = D[0] / h[0] * (u_n[1] - u_n[0])
            - h[0] / 2.
                * ((u_n[0] - u_nm1[0]) / dt + a * (u_n[1] - u_n[0]) / h[0] + c * u_n[0] - f[0]);

        (u_n, u_interface, phi_interface)
    }
}

impl Discretization for Volumes {
    #[allow(non_snake_case)]
    fn precompute_Y(
        M: usize,
        h: &ArrayView1<f64>,
        D: &ArrayView1<f64>,
        a: f64,
        c: f64,
        dt: f64,
        _f: &ArrayView1<f64>,
        _bd_cond: f64,
        Lambda: f64,
        upper_domain: bool,
    ) -> [Array1<f64>; 3] {
        let h = flip_if(!upper_domain, *h);
        let D = flip_if(!upper_domain, *D);
        volumes::get_Y(M, &h, &D, a, c, dt, Lambda, upper_domain)
    }

    #[allow(non_snake_case)]
    fn integrate_one_step(
        M: usize,
        h: &ArrayView1<f64>,
        D: &ArrayView1<f64>,
        a: f64,
        c: f64,
        dt: f64,
        f: &ArrayView1<f64>,
        bd_cond: f64,
        Lambda: f64,
        u_nm1: &ArrayView1<f64>,
        u_interface: f64,
        phi_interface: f64,
        Y: &[Array1<f64>; 3],
        upper_domain: bool,
    ) -> (Array1<f64>, f64, f64) {
        assert!(dt > 0.);
        assert_eq!(D.len(), M + 1);
        assert_eq!(h.len(), M);
        assert_eq!(f.len(), M);
        let f = flip_if(!upper_domain, *f);
        let h = flip_if(!upper_domain, *h);
        let D = flip_if(!upper_domain, *D);
        let u_nm1 = flip_if(!upper_domain, *u_nm1);

        let rhs = (diff(&f) + diff(&u_nm1) / dt) * dt / (1. + dt * c);
        let cond_0;
        let cond_M;
        if upper_domain {
            // Neumann condition: user give derivative but I need flux

            let cond_robin = Lambda * u_interface + phi_interface
                - Lambda * dt / (1. + dt * c) * (f[0] + u_nm1[0] / dt);

            cond_M = bd_cond * D[D.len() - 1];
            cond_0 = cond_robin;
        } else {
            // Dirichlet condition: user gives value, rhs is more complicated
            let cond_robin = Lambda * u_interface + phi_interface
                - Lambda * dt / (1. + dt * c) * (f[f.len() - 1] + u_nm1[u_nm1.len() - 1] / dt);

            cond_M = cond_robin;
            cond_0 = bd_cond - dt / (1. + dt * c) * (f[0] + u_nm1[0] / dt);
        }
        let rhs = stack![Axis(0), array![cond_0], rhs, array![cond_M]];

        let phi_ret = solve_linear([&Y[0].view(), &Y[1].view(), &Y[2].view()], &rhs.view());
        let d = &phi_ret / &D; // We take the derivative of u

        let d_kp1 = d.slice(s![1..]);
        let d_km1 = d.slice(s![..d.len() - 1]);

        let u_n = (&f + &(&u_nm1 / dt) + &diff(&(&D * &d)) / &h - (&d_kp1 + &d_km1) * (a / 2.))
            * dt
            / (1. + dt * c);

        debug_assert_eq!(u_n.len(), M);
        let u_phi = match upper_domain {
            true => (u_n[0] - h[0] * d[1] / 6. - h[0] * d[0] / 3., phi_ret[0]),
            false => (
                u_n[u_n.len() - 1]
                    + h[h.len() - 1] * d[d.len() - 2] / 6.
                    + h[h.len() - 1] * d[d.len() - 1] / 3.,
                phi_ret[phi_ret.len() - 1],
            ),
        };

        let u_n = flip_if(!upper_domain, u_n.view()).to_owned();
        (u_n, u_phi.0, u_phi.1)
    }
}

#[cfg(test)]
mod tests;
