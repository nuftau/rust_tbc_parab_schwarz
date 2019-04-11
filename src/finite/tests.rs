
#[allow(non_snake_case)]
#[test]
fn test_finite_differences_schwarz() {
    use crate::utils::linalg::{diff, cos, sin, norm};
    use super::Differences;
    use super::Discretization;
    use super::differences;
    use ndarray::{Array, Axis, stack};
    let a = 1.;
    let c = 0.3;

    let T = 5.;
    let d = 8.;
    let t = 3.;
    let dt = 0.05;
    let M1 = 10;
    let M2 = 10;

    let x1 = Array::linspace(0., -1., M1).mapv(|a:f64| a.powi(3));
    let x2 = Array::linspace(0., 1., M2).mapv(|a:f64| a.powi(4));

    let h1 = diff(&x1.view());
    let h2 = diff(&x2.view());

    // coordinates at half-points:
    let x1_1_2 = &x1.slice(s![..x1.len()-1]) + &(&h1 / 2.);
    let x2_1_2 = &x2.slice(s![..x2.len()-1]) + &(&h2 / 2.);

    let D1 = &x1_1_2.mapv(|a:f64| a.powi(2)) + 1.2;
    let D2 = &x2_1_2.mapv(|a:f64| a.powi(2)) + 2.2;

    let D1_x = &x1*&x1 + 1.2;
    let D2_x = &x2*&x2 + 2.2;
    let D1_prime = &x1*2.;
    let D2_prime = &x2*2.;

    let ratio_D = D1_x[0] / D2_x[0];

    let t_n = t;
    let t = t + dt;
    let neumann = ratio_D * d*f64::cos(x2_1_2[x2_1_2.len()-1]*d);
    let dirichlet = f64::sin(x1[x1.len()-1]*d) + T*t;

    // Note: f is a local approximation !
    let f2 = T*(1.+c*t) + ratio_D * (d*a*cos(&(&x2*d)) + c*sin(&(&x2*d))
            + &D2_x * d*d *sin(&(&x2*d)) - D2_prime * d * cos(&(&x2*d)));

    let f1 = T*(1.+c*t) + d*a*cos(&(&x1*d)) + c*sin(&(&x1*d))
            + &D1_x * d*d *sin(&(&x1*d)) - D1_prime * d * cos(&(&x1*d));

    let slice_x1 = x1.slice(s![..;-1]);
    let slice_cut_x1 = slice_x1.slice(s![..x1.len()-1]);
    let u1_0 = sin(&(&slice_cut_x1*d)) + T*t_n;
    let u2_0 = ratio_D * sin(&(&x2*d)) + T*t_n;

    let u0 = stack(Axis(0),&[u1_0.view(), u2_0.view()])
                .expect("cannot stack u1_0 and u2_0");

    //TODO the difference with python comes from here:
    //in python code we dont take the last D value
    //and we use the interface one.
    //D1 = np.concatenate(([D1_x[0]], D1[:-1]))
    //D2 = np.concatenate(([D2_x[0]], D2[:-1]))
    let ret = differences::integrate_one_step_star(M1, M2, &h1.view(), &h2.view(), &D1.view(), &D2.view(),
                                             a, c, dt, &f1.view(), &f2.view(),
                                             neumann, dirichlet, &u0.view());
    // ret = u_np1, real_u_interface, real_phi_interface
    let u_np1 = ret.0;
    let real_u_interface = ret.1;
    let real_phi_interface = ret.2;

    // Schwarz paramters:
    let Lambda_1 = 15.;
    let Lambda_2 = 0.3;
    let u1_0 = sin(&(&x1*d)) + T*t_n;
    let u2_0 = ratio_D * sin(&(&x2*d)) + T*t_n;

    let mut u_interface = real_u_interface;
    let mut phi_interface = real_phi_interface;

    let mut ecart = 0.;
    for _i in 0..30 {
        let Y = Differences::precompute_Y(M2, &h2.view(), &D2.view(), a, c,
                                                  dt, &f2.view(), neumann,
                                                  Lambda_2, true);
        let ret = Differences::integrate_one_step(M2, &h2.view(), &D2.view(), a, c,
                                                  dt, &f2.view(), neumann,
                                                  Lambda_2, &u2_0.view(),
                                                  u_interface,
                                                  phi_interface, &Y, true);
        let u2_1 = ret.0;
        u_interface = ret.1;
        phi_interface = ret.2;
        let Y = Differences::precompute_Y(M1, &h1.view(), &D1.view(), a, c,
                                                  dt, &f1.view(), dirichlet,
                                                  Lambda_1, false);
        let ret = Differences::integrate_one_step(M1, &h1.view(), &D1.view(), a, c,
                                                  dt, &f1.view(), dirichlet,
                                                  Lambda_1, &u1_0.view(),
                                                  u_interface,
                                                  phi_interface, &Y, false);
        let u1_1 = ret.0;
        u_interface = ret.1;
        phi_interface = ret.2;

        let slice_u1 = u1_1.slice(s![..;-1]);
        let slice_cut_u1 = slice_u1.slice(s![..u1_1.len()-1]);
        let u_np1_schwarz = stack(Axis(0),&[slice_cut_u1.view(),
                                            u2_1.view()])
                    .expect("cannot stack u1_1 and u2_1");
        ecart = norm(&(&u_np1 - &u_np1_schwarz));
    }
    println!("{}", ecart);
    assert!(ecart < 3e-11);
}

#[allow(non_snake_case)]
#[test]
fn test_finite_differences_star() {
    use crate::utils::linalg::{diff, cos, sin, norm};
    use super::differences;
    use ndarray::{Array, Axis, stack};
    let a = 1.2;
    let c = 0.3;

    let T = 5.;
    let d = 8.;
    let t = 3.;
    let dt = 0.05;
    let M1 = 3000;
    let M2 = 3000;

    let x1 = Array::linspace(0., -1., M1);
    let x2 = Array::linspace(0., 1., M2);

    let h1 = diff(&x1);
    let h2 = diff(&x2);

    // coordinates at half-points:
    let x1_1_2 = &x1.slice(s![..x1.len()-1]) + &(&h1 / 2.);
    let x2_1_2 = &x2.slice(s![..x2.len()-1]) + &(&h2 / 2.);

    let D1 = &x1_1_2.mapv(|a| a.powi(2)) + 1.2;
    let D2 = &x2_1_2.mapv(|a| a.powi(2)) + 2.2;

    let D1_x = &x1*&x1 + 1.2;
    let D2_x = &x2*&x2 + 2.2;
    let D1_prime = &x1*2.;
    let D2_prime = &x2*2.;

    let ratio_D = D1_x[0] / D2_x[0];

    let t_n = t;
    let t = t + dt;
    let neumann = ratio_D * d*f64::cos(x2_1_2[x2_1_2.len()-1]*d);
    let dirichlet = f64::sin(x1[x1.len()-1]*d) + T*t;

    // Note: f is a local approximation !
    let f2 = T*(1.+c*t) + ratio_D * (d*a*cos(&(&x2*d)) + c*sin(&(&x2*d))
            + &D2_x * d*d *sin(&(&x2*d)) - D2_prime * d * cos(&(&x2*d)));

    let f1 = T*(1.+c*t) + d*a*cos(&(&x1*d)) + c*sin(&(&x1*d))
            + &D1_x * d*d *sin(&(&x1*d)) - D1_prime * d * cos(&(&x1*d));

    let slice_x1 = x1.slice(s![..;-1]);
    let slice_cut_x1 = slice_x1.slice(s![..x1.len()-1]);
    let u0_1 = sin(&(&slice_cut_x1*d)) + T*t_n;
    let u0_2 = ratio_D * sin(&(&x2*d)) + T*t_n;

    let u0 = stack(Axis(0),&[u0_1.view(), u0_2.view()])
                .expect("cannot stack u1_0 and u2_0");

    let u1 = stack(Axis(0), &[(sin(&(&slice_cut_x1*d)) + T*t).view(),
                   (ratio_D * sin(&(&x2*d)) + T*t).view()])
                .expect("cannot stack u1_1 and u2_1");

    let ret = differences::integrate_one_step_star(M1, M2, &h1.view(), &h2.view(), &D1.view(), &D2.view(),
                                             a, c, dt, &f1.view(), &f2.view(),
                                             neumann, dirichlet, &u0.view());
    // ret = u_np1, real_u_interface, real_phi_interface
    let u_np1 = ret.0;
    let error = norm(&(&u1-&u_np1));
    assert!(error < 3e-5);
}
