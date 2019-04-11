from target.release import librust_rate_constant
print(librust_rate_constant.rate(time_window_len=10,
           Lambda_1=3., Lambda_2=0.,
           a=.0, c=0., dt=.1,
           M1=400, M2=400, h1=-.5, h2=.5, D1=.6, D2=.54,
           is_finite_differences=True,
           number_samples=10))
print(librust_rate_constant.rate(time_window_len=10,
           Lambda_1=3., Lambda_2=0.,
           a=.0, c=0., dt=.1,
           M1=400, M2=400, h1=.5, h2=.5, D1=.6, D2=.54,
           is_finite_differences=False,
           number_samples=10))
