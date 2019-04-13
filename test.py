#!/usr/bin/python3
"""
    This module is for testing purpose only.
    It is mainly a copy-paste of rust_mod in the main Python repository,
    but there are differences
"""
from cffi import FFI
import numpy as np
import os
import sys
os.system('cargo build --release')

ffi = FFI()
# TODO remove or adapt
ffi.cdef("""
    double rate(unsigned long time_window_len, double Lambda_1, double Lambda_2, double a, double c, double dt, unsigned long M1, unsigned long M2,
        const double* h1, const double* h2, double* D1, double* D2, int is_finite_differences, unsigned long number_samples);
    const double* interface_err(unsigned long time_window_len, double Lambda_1, double Lambda_2, double a, double c, double dt, unsigned long M1, unsigned long M2,
        const double* h1, const double* h2, double* D1, double* D2, int is_finite_differences, unsigned long number_samples);
    const double* full_interface_err(unsigned long time_window_len, double Lambda_1, double Lambda_2, double a, double c, double dt, unsigned long M1, unsigned long M2,
        const double* h1, const double* h2, double* D1, double* D2, int is_finite_differences, unsigned long number_samples);
    """)

# pylint: disable=invalid-name

def _as_f64_array(array):
    return ffi.cast('const double *', array.ctypes.data)


def _as_f64(num):
    """ Cast np.float64 for Rust."""
    return ffi.cast("double", num)


def _as_u64(num):
    """ Cast `num` to Rust `usize`."""
    return ffi.cast("unsigned long", num)


def bool_as_i32(num):
    """ Cast `num` to Rust `usize`."""
    if num:
        ret = 1
    else:
        ret = 0
    return ffi.cast("int", ret)


# Go get the Rust library.
# TODO must compute rate = mean(errors[2])/mean(errors[1])
# right now it is mean(errors[2]/errors[1])
lib = ffi.dlopen(
    "target/release/librust_rate_constant.so")

def errors_raw(is_finite_differences,
               N,
               Lambda_1,
               Lambda_2,
               a,
               c,
               dt,
               M1,
               M2,
               number_seeds=10,
               function_D1=lambda x: .6+np.zeros_like(x),
               function_D2=lambda x: .54+np.zeros_like(x)):
    size_domain_1 = 200
    size_domain_2 = 200
    if is_finite_differences:
        x1 = -np.linspace(0, size_domain_1, M1)**1
        x2 = np.linspace(0, size_domain_2, M2)**1
        h1 = np.diff(x1)
        h2 = np.diff(x2)
        # coordinates at half-points:
        x1_1_2 = x1[:-1] + h1 / 2
        x2_1_2 = x2[:-1] + h2 / 2
        D1 = function_D1(x1_1_2)
        D2 = function_D2(x2_1_2)
    else:
        h1 = size_domain_1 / M1 + np.zeros(M1)
        h2 = size_domain_2 / M2 + np.zeros(M2)
        x1_1_2 = np.cumsum(np.concatenate(([0], h1)))
        x2_1_2 = np.cumsum(np.concatenate(([0], h2)))

        D1 = function_D1(x1_1_2)
        D2 = function_D2(x2_1_2)

    time_window_len = _as_u64(N)
    Lambda_1arg = _as_f64(Lambda_1)
    Lambda_2arg = _as_f64(Lambda_2)
    aarg = _as_f64(a)
    carg = _as_f64(c)
    dtarg = _as_f64(dt)
    M1arg = _as_u64(M1)
    M2arg = _as_u64(M2)
    number_samples = _as_u64(number_seeds)

    h1arg, h2arg, D1arg, D2arg = _as_f64_array(h1), _as_f64_array(h2), \
        _as_f64_array(D1), _as_f64_array(D2)

    ptr = lib.full_interface_err(time_window_len, Lambda_1arg, Lambda_2arg,
                                 aarg, carg, dtarg, M1arg, M2arg, h1arg, h2arg,
                                 D1arg, D2arg, is_finite_differences,
                                 number_samples)
    buf_ret = np.reshape(
        np.frombuffer(ffi.buffer(ptr, 8 * N * 3 * number_seeds),
                      dtype=np.float64), (number_seeds, 3, N))
    return buf_ret


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("ok. let's get some values")
        print(errors_raw(is_finite_differences=True,
                   N=1000,
                   Lambda_1=.6,
                   Lambda_2=0.,
                   a=0.,
                   c=1e-10,
                   dt=.1,
                   M1=2000,
                   M2=2000,
                   number_seeds=10))
    elif sys.argv[1] == "plotfft":
        import matplotlib.pyplot as plt
        errors_ret = errors_raw(is_finite_differences=True,
                   N=1000,
                   Lambda_1=.6,
                   Lambda_2=0.,
                   a=0.,
                   c=1e-10,
                   dt=.1,
                   M1=200,
                   M2=200,
                   number_seeds=50)
        plt.plot(np.mean(np.abs(np.fft.fftshift(np.fft.fft(errors_ret[:,0],
            norm="ortho", axis=-1), axes=(-1,))), axis=0))
        plt.show()

