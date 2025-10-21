
# ================================================================
#  main.py
#  ---------------------------------------------------------------
#  purpose:
#      run a line-search optimizer on the given dataset (FFD.csv) to fit
#      a damped, chirped sinusoid:
#          W(t) = A0 + A * exp(-t/tau) * sin((omega + alpha*t)*t + phi)
#      with parameters x = [A0, A, tau, omega, alpha, phi].
#
#  usage:
#      python main.py
#
#  inputs:
#      - FFD.csv (two columns: t, v)
#
#  outputs (saved under ./results_Project1/):
#      - iteration logs and summaries (produced by the optimizer utilities)
#      - initial_guess.txt (the initial x0 used for the run)
# ================================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

from run_logger import RunLogger

workspace_root = Path(__file__).resolve().parent.parent   
sys.path.insert(0, str(workspace_root))

import functions as fn 

# ----------------------------------------------------------------
# User Settings
# ----------------------------------------------------------------

# data and outputs
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file  = os.path.join(script_dir, "FFD.csv")
out_dir    = os.path.join(script_dir, "results_Project1")
os.makedirs(out_dir, exist_ok = True)

# settings
alpha_bar  = 1.0
c1         = 0.001
c2         = 0.5
rho        = 0.5

# initial-guess overrides (set to None to use auto estimation)
init_a0     = None
init_a      = None
init_tau    = None
init_omega  = None
init_alpha  = None
init_phi    = None


# line search options
line_search_dict = {
    "armijo": (
        fn.armijo_backtracking,
        dict(alpha_bar = alpha_bar, c1 = c1, rho = rho),
    ),
    "strong_wolfe": (
        fn.strong_wolfe,
        dict(alpha_bar = alpha_bar, c1 = c1, c2 = c2),
    ),
}

# choose line search
line_search = "strong_wolfe"

# optional extra overrides from user settings (can be {} if unused)
line_search_options_extra = {}
line_search_options_extra = line_search_options_extra if 'LS_EXTRA_OPTS' in globals() else {}

# optimizer options
optimizer_dict = {
    "gd":   fn.gradient_descent,
    "cgd":  fn.conjugate_gradient_descent,
    "bfgs": fn.quasi_newton_bfgs,
}

# choose optimizer
optimizer = "bfgs"


# ----------------------------------------------------------------
#  settings for optimization
# ----------------------------------------------------------------

# function wrapper
def f_wrapper(x):
    f, _ = objective_function_and_gradient(x, t, v)
    return float(f)

# gradient wrapper
def g_wrapper(x):
    _, g = objective_function_and_gradient(x, t, v)
    return np.asarray(g, dtype=float)

# line search selection based on user choice
ls_func, ls_options = line_search_dict[line_search]

# optimizer selection based on user choice
opt_func = optimizer_dict[optimizer]

# logger
run_tag = "project01"

# ----------------------------------------------------------------
#  objective function and gradient
# ----------------------------------------------------------------

def objective_function_and_gradient(x, tk, v):
    """
    compute least-squares objective f(x) = sum((W - v)^2)

    Given function: 
        W(t) = A0 + A * exp(-t / tau) * sin((omega + alpha * t) * t + phi)
        
    parameters:
        x = [A0, A, tau, omega, alpha, phi]

    inputs:
        tk = time samples, 1D array
        v  = observed data, 1D array

    returns:
        f = objective function value
        grad = gradient, 1D array of same length as x

    """
    # parameters
    a0, a, tau, w, alpha, phi = x

    phase = (w + alpha * tk) * tk + phi
    E = np.exp(-tk / tau)
    Sk = E * np.sin(phase)
    Ck = E *  np.cos(phase)

    # model
    W = a0 + a * Sk

    # residual
    r = W - v

    # objective function value
    f = np.sum(r ** 2)

    # gradient
    grad = 2 * np.array([
        np.sum(r * 1.0),                        # df/dA0
        np.sum(r * Sk),                         # df/dA
        np.sum(r * (a * Sk * (tk / tau**2))),   # df/dtau
        np.sum(r * (a * tk * Ck)),              # df/domega
        np.sum(r * (a * (tk**2) * Ck)),         # df/dalpha
        np.sum(r * (a * Ck))                    # df/dphi
    ])

    return f, grad


# ----------------------------------------------------------------
#  initial guess estimation
# ----------------------------------------------------------------

def estimate_initial_guess(t, v):
    """
    Initial guesses for [A0, A, tau, omega, alpha, phi] from (t, v).

    Method:
      - a0   : Mean of last 10% of samples.
      - a    : Max of first 10% minus a0.
      - tau  : Decay from first/last peak of |v - a0|.
      - omega: Period from first two maxima.
      - alpha: Slope of freq vs. peak time.
      - phi  : Phase from projections onto decay sin/cos.

    """

    # prepare
    t = np.asarray(t); v = np.asarray(v)
    order = np.argsort(t)
    t_sorted = t[order]
    v_sorted = v[order]

    # a0 (dc offset)
    tail_frac = 0.10
    n_tail = max(1, int(np.ceil(v_sorted.size * tail_frac)))
    init_a0 = float(np.mean(v_sorted[-n_tail:]))

    # a (amplitude near t=0)
    head_frac = 0.10
    n_head = max(1, int(np.ceil(v_sorted.size * head_frac)))
    init_a = float(np.max(v_sorted[:n_head]) - init_a0)
    if init_a <= 0:
        init_a = abs(init_a)

    # residual and simple peak indices
    y = v_sorted - init_a0
    a = np.abs(y)
    # peaks of |y| (for decay)
    pk_abs = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]))[0] + 1
    # peaks of y itself (for period)
    pk_y = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

    # tau (decay time)
    if pk_abs.size >= 2:
        p1 = max(a[pk_abs[0]], 1e-12)
        pL = max(a[pk_abs[-1]], 1e-12)
        ratio = max(p1 / pL, 1 + 1e-6)  # ensure log > 0
        init_tau = float((t_sorted[pk_abs[-1]] - t_sorted[pk_abs[0]]) / np.log(ratio))
        dt = max(1e-12, float(np.min(np.diff(t_sorted))))
        init_tau = max(init_tau, dt)
    else:
        init_tau = float(t_sorted[-1] - t_sorted[0])  

    # omega (base angular freq)
    if pk_y.size >= 2:
        T = float(t_sorted[pk_y[1]] - t_sorted[pk_y[0]])
        init_omega = float(2.0 * np.pi / max(T, 1e-12))
    else:
        init_omega = float(2.0 * np.pi / max(t_sorted[-1] - t_sorted[0], 1e-12))

    # alpha (chirp)
    init_alpha = 0.0 
    if pk_y.size >= 2:
        peak_times = t_sorted[pk_y]
        periods = np.diff(peak_times)
        freqs = 1.0 / np.maximum(periods, 1e-12)
        freq_times = peak_times[1:]

        if freq_times.size > 1:
            coeffs = np.polyfit(freq_times, freqs, 1)
            init_alpha = coeffs[0]
        else:
            init_alpha = 0.0  


    # phi (phase)
    t0 = 2.0 * np.pi / max(init_omega, 1e-12)
    t_end = t_sorted[0] + 2.0 * t0
    mask = t_sorted <= t_end
    if not np.any(mask):
        mask = np.ones_like(t_sorted, dtype = bool)

    tt = t_sorted[mask]
    yy = y[mask]
    theta = (init_omega + init_alpha * tt) * tt    
    E = np.exp(-tt / max(init_tau, 1e-12))
    S = float(np.dot(yy, E * np.sin(theta)))  
    C = float(np.dot(yy, E * np.cos(theta)))  
    init_phi = float(np.arctan2(C, S))

    return np.array([init_a0, init_a, init_tau, init_omega, init_alpha, init_phi], dtype = float)


     
# ----------------------------------------------------------------
#  load data
# ----------------------------------------------------------------

# give error if data file not found
if not os.path.exists(data_file):
    raise FileNotFoundError(f"data file not found: {data_file}")

# read data 
df = pd.read_csv(data_file, header = None, usecols = [0,1], names = ["t","v"])
t = df["t"].to_numpy(dtype = float)
v = df["v"].to_numpy(dtype = float)


# ---------------------------------------------------------------
# run optimization
# ---------------------------------------------------------------


# initial guess: use overrides if set, else auto
auto_a0, auto_a, auto_tau, auto_omega, auto_alpha, auto_phi = estimate_initial_guess(t, v)

a0    = init_a0    if init_a0    is not None else auto_a0
a     = init_a     if init_a     is not None else auto_a
tau   = init_tau   if init_tau   is not None else auto_tau
omega = init_omega if init_omega is not None else auto_omega
alpha = init_alpha if init_alpha is not None else auto_alpha
phi   = init_phi   if init_phi   is not None else auto_phi

x0 = np.array([a0, a, tau, omega, alpha, phi], dtype = float)

# save the initial guess used
with open(os.path.join(out_dir, f"{optimizer}_{line_search}initial_guess.txt"), "w") as fh:
    fh.write("x0 = [A0, A, tau, omega, alpha, phi]\n")
    fh.write(np.array2string(x0, precision=7, separator=", ") + "\n")

# optimizer options
opts = {
    "max_iter": 1000,
    "tol": 1e-6,
    "line_search_opts": ls_options,   
    "save_flag": True,
    "optimizer": optimizer,
    "line_search": line_search,
    "out_dir": out_dir,
    "run_tag": run_tag,
    "alpha0": alpha_bar,
    "c1": c1,
    "c2": c2,   
}

# run 
result = opt_func(f_wrapper, g_wrapper, x0, ls_func, opts=opts)

# print results
print("success:", result["success"])
print("iters  :", result["n_iter"])
print("f_final:", result["f"])
print("x_final:", result["x"])
