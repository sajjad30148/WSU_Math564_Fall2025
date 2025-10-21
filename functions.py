# ================================================================
#  functions.py
#  ---------------------------------------------------------------
#  purpose:
#      provide line-search methods and optimizers.
#
#  inputs expected by callers:
#      - f(x): scalar objective
#      - grad(x): gradient vector
#      - x0: initial parameter vector
#      - line_search: one of {armijo_backtracking, strong_wolfe}
#      - opts: dict with keys such as:
#          max_iter, tol, line_search_opts, save_flag,
#          optimizer, line_search, out_dir, run_tag, parameters for line search
#
#  outputs:
#      each optimizer returns a dict:
#          {"x", "f", "g", "n_iter", "n_func_eval", "n_grad_eval", "success"}
# ================================================================


import numpy as np  
import pandas as pd 
from run_logger import RunLogger


# ----------------------------------------------------------------
# line search methods
# ----------------------------------------------------------------

def armijo_backtracking(f, grad, xk, pk, alpha_bar=1.0, c1=1e-3, rho=0.5,
                       max_backtracks=50, min_alpha=None, **kwargs):
    """
    Armijo backtracking line search (simple, safe).

    Finds alpha > 0 s.t.
        f(xk + alpha*pk) <= f(xk) + c1 * alpha * grad(xk)^T pk
    by shrinking alpha <- rho * alpha.

    Inputs
        f, grad   : callables (f(x)->scalar, grad(x)->vector)
        xk        : current point
        pk        : descent direction (requires grad(xk)^T pk < 0)
        alpha_bar : initial trial step
        c1        : Armijo parameter in (0, 1)
        rho       : shrink factor in (0, 1)
        max_backtracks : cap on shrink steps
        min_alpha : optional floor for alpha (default: machine eps)

    Returns
        alpha     : accepted step
        f_new     : f(xk + alpha*pk)
        n_eval    : number of f-evals during the search (excludes f(xk))
    """
    # --- basic checks ---
    if not (0.0 < c1 < 1.0):
        raise ValueError("c1 must be in (0, 1).")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1).")

    xk = np.asarray(xk, dtype=float)
    pk = np.asarray(pk, dtype=float)

    fk = float(f(xk))
    gk = np.asarray(grad(xk), dtype=float)
    slope0 = float(np.dot(gk, pk))
    if not np.isfinite(slope0):
        raise ValueError("non-finite directional derivative at xk")
    if slope0 >= 0.0:
        raise ValueError("pk is not a descent direction (grad^T pk >= 0)")

    alpha = float(alpha_bar)
    n_eval = 0
    alpha_floor = np.finfo(float).eps if min_alpha is None else float(min_alpha)

    # --- backtracking loop ---
    backtrack_count = 0
    while True:
        x_trial = xk + alpha * pk
        f_trial = float(f(x_trial))
        n_eval += 1

        # if f_trial is not finite, keep shrinking until finite or floor
        while not np.isfinite(f_trial) and alpha > alpha_floor and backtrack_count < int(max_backtracks):
            alpha *= rho
            backtrack_count += 1
            x_trial = xk + alpha * pk
            f_trial = float(f(x_trial))
            n_eval += 1

        # Armijo sufficient decrease
        if np.isfinite(f_trial) and f_trial <= fk + c1 * alpha * slope0:
            return alpha, f_trial, n_eval

        # shrink step
        alpha *= rho
        backtrack_count += 1

        # stop if hit floor or backtrack cap
        if (alpha <= alpha_floor) or (backtrack_count >= int(max_backtracks)):
            # return best effort at current (possibly shrunk) alpha
            x_trial = xk + alpha * pk
            f_trial = float(f(x_trial))
            n_eval += 1
            return alpha, f_trial, n_eval


        

def strong_wolfe(f, grad, xk, pk, alpha_bar = 1.0, c1 = 1e-4, c2 = 0.9):    
    """
    strong wolfe line search

    purpose
        find a step length alpha > 0 that satisfies the strong wolfe conditions
        along a given descent direction pk at the current point xk

    conditions enforced
        armijo sufficient descent:
            f(xk + alpha pk) <= f(xk) + c1 * alpha * grad(xk)^T pk
        strong curvature:
            abs(grad(xk + alpha pk)^T pk) <= -c2 * grad(xk)^T pk
        with 0 < c1 < c2 < 1

    inputs
        f              callable, f(x) -> scalar objective value
        grad           callable, grad(x) -> gradient vector at x  
        xk             current point  
        pk             search direction at xk, should satisfy grad(xk)^T pk < 0
        alpha_bar      initial trial step length, default 1.0
        c1             armijo parameter in (0, 1), default 0.001
        c2             curvature parameter in (c1, 1), default 0.5

    returns
        alpha          accepted step length satisfying strong wolfe, or a fallback if limits are hit
        f_new          objective value at xk + alpha pk
        n_eval         number of objective evaluations performed inside the line search

    notes
        this function evaluates both f and grad at trial points
        only the count of f evaluations is returned as n_eval for consistency with armijo_backtracking

    references
        nocedal and wright, numerical optimization, algorithms 3.5 and 3.6
    """

    # evaluate objective, gradient and search direction at current point
    xk = np.asarray(xk, dtype = float)
    pk = np.asarray(pk, dtype = float)
    fk = float(f(xk))
    gk = np.asarray(grad(xk), dtype = float)

    # initial directional derivative (grad^T pk)
    slope0 = float(np.dot(gk, pk))

    if not np.isfinite(fk) or not np.all(np.isfinite(gk)): # guard: directional derivative must be finite
        raise ValueError("non finite fk or grad(xk)")
    
    if slope0 >= 0.0: # guard: pk must be a descent direction (grad^T pk < 0)
        raise ValueError("pk is not a descent direction at xk since grad(xk)^T pk >= 0")

    # helper subfunctions for phi: phi(alpha) = f(xk + alpha pk)
    def phi(alpha):
        return float(f(xk + alpha * pk))

    # helper subfunctions for derivative of phi: dphi(alpha) = grad(xk+alpha pk)^T pk
    def dphi(alpha):
        return float(np.dot(grad(xk + alpha * pk), pk))

    n_eval = 0 # to count of objective evaluations performed inside line search

    # phase a: bracketing as in algorithm 3.5
    alpha_prev = 0.0
    f_prev = fk
    alpha = float(alpha_bar)
    i = 0

    while True:
        f_current = phi(alpha)
        n_eval += 1

        # check armijo condition, modify bracket and enter zoom phase if satisfied
        if (f_current > fk + c1 * alpha * slope0) or (i > 0 and f_current >= f_prev):
            alpha_lo = alpha_prev
            alpha_hi = alpha
            f_lo = f_prev
            f_hi = f_current
            break

        # compute directional derivative at current alpha
        slope_alpha = dphi(alpha)

        # check strong wolfe curvature condition, if satisfied return alpha
        if abs(slope_alpha) <= c2 * abs(slope0):
            return alpha, f_current, n_eval
        
        # check for too large step (slope is nonnegative), modify bracket and enter zoom phase if so
        if slope_alpha >= 0.0:
            alpha_lo = alpha
            alpha_hi = alpha_prev
            f_lo = f_current
            f_hi = f_prev
            break

        # otherwise, expand the step and continue the bracketing phase
        alpha_prev = alpha
        f_prev = f_current
        alpha = 2.0 * alpha  # double the step
        i += 1

    # phase b: zoom as in algorithm 3.6
    while True:

        # bisection to find a trial alpha in (alpha_lo, alpha_hi)
        alpha_j = 0.5 * (alpha_lo + alpha_hi)

        # evaluate function at trial alpha_j
        f_j = phi(alpha_j)
        n_eval += 1

        # Armijo condition check inside zoom (use fk + c1 * alpha_j * slope0)
        if (f_j > fk + c1 * alpha_j * slope0) or (f_j >= f_lo):
            alpha_hi = alpha_j
            continue

        # evaluate derivative at trial alpha_j
        slope_j = dphi(alpha_j)

        # check strong wolfe curvature condition inside zoom
        if abs(slope_j) <= c2 * abs(slope0):
            return alpha_j, f_j, n_eval
        
        # sign test to decide which side to continue the search
        if slope_j * (alpha_hi - alpha_lo) >= 0.0:
            alpha_hi = alpha_lo

        # move the lower bracket to alpha_j since Armijo passed and curvature failed
        alpha_lo = alpha_j
        f_lo = f_j

    # If the loop exits unexpectedly, return the low end of the bracket which satisfies Armijo
    f_star = phi(alpha_lo)
    n_eval += 1

    return alpha_lo, f_star, n_eval
      


# ----------------------------------------------------------------
# optimization drivers
# ----------------------------------------------------------------

def gradient_descent(f, grad, x0, line_search, opts=None):
    """
    Gradient descent with line search (simple, safe).

    Returns a dict with: x, f, g, n_iter, n_func_eval, n_grad_eval, success
    """
    opts = {} if opts is None else dict(opts)

    # --- user options ---
    max_iter  = int(opts.get('max_iter', 1000))
    tol       = float(opts.get('tol', 1e-6))
    ls_opts   = dict(opts.get('line_search_opts', {}))   # will be merged with meta defaults
    save_flag = bool(opts.get('save_flag', True))

    optimizer = str(opts.get('optimizer', 'gradient_descent'))
    ls_name   = str(opts.get('line_search', 'armijo_backtracking'))
    out_dir   = str(opts.get('out_dir', './results'))
    run_tag   = str(opts.get('run_tag', 'None'))

    # meta defaults for common line searches
    meta_alpha0 = float(opts.get('alpha0', 1.0))
    meta_c1     = float(opts.get('c1', ls_opts.get('c1', 1e-3)))
    meta_c2     = float(opts.get('c2', ls_opts.get('c2', 0.5)))  # used by strong-wolfe

    # merge meta defaults only if not provided explicitly
    ls_opts.setdefault('alpha_bar', meta_alpha0)
    ls_opts.setdefault('c1', meta_c1)
    # c2 is harmless for Armijo; strong-wolfe will use it
    ls_opts.setdefault('c2', meta_c2)

    # tiny movement floor (prevents endless micro-steps)
    step_floor = float(opts.get('step_floor', 1e-16))

    # --- optional logger ---
    logger = None
    if save_flag:
        try:
            logger = RunLogger(out_dir=out_dir,
                               optimizer=optimizer,
                               line_search=ls_name,
                               alpha0=meta_alpha0, c1=meta_c1, c2=meta_c2,
                               max_iter=max_iter, gtol=tol, run_tag=run_tag)
        except NameError:
            # if RunLogger is not defined, disable logging quietly
            logger = None
            save_flag = False

    # --- initialization ---
    xk = np.asarray(x0, dtype=float)
    fk = float(f(xk))
    gk = np.asarray(grad(xk), dtype=float)

    n_func_eval = 1  # f(x0)
    n_grad_eval = 1  # grad(x0)

    for k in range(1, max_iter + 1):
        gk_norm = float(np.linalg.norm(gk, ord=2))
        if gk_norm <= tol:
            success = True
            if logger is not None:
                logger.add_eval_counts(f_evals=n_func_eval - 1, g_evals=n_grad_eval - 1)
                logger.finalize(success=success, n_iter=k - 1, f_final=fk,
                                g_final=gk_norm, x_final=xk)
                logger.close()
            return {"x": xk, "f": fk, "g": gk, "n_iter": k - 1,
                    "n_func_eval": n_func_eval, "n_grad_eval": n_grad_eval,
                    "success": success}

        pk = -gk
        p_norm = float(np.linalg.norm(pk, ord=2))

        # line search (alpha, f_new, n_eval_f)
        alpha, f_new, n_eval_f = line_search(f, grad, xk, pk, **ls_opts)
        n_func_eval += int(n_eval_f)

        # guard: non-finite trial or microscopic step
        if not np.isfinite(f_new) or alpha * p_norm <= step_floor:
            success = False
            if logger:
                logger.add_eval_counts(f_evals=n_eval_f)
                logger.finalize(success=success, n_iter=k - 1, f_final=fk,
                                g_final=gk_norm, x_final=xk)
                logger.close()
            return {"x": xk, "f": fk, "g": gk, "n_iter": k - 1,
                    "n_func_eval": n_func_eval, "n_grad_eval": n_grad_eval,
                    "success": success}

        # take step
        xk_next = xk + alpha * pk
        fk_next = float(f_new)
        gk_next = np.asarray(grad(xk_next), dtype=float)
        n_grad_eval += 1

        if logger:
            df_abs = float(abs(fk - fk_next))
            logger.add_eval_counts(f_evals=n_eval_f)
            logger.log_iter(int(k), float(fk_next),
                            float(np.linalg.norm(gk_next, 2)),
                            float(alpha), float(p_norm), float(df_abs))

        # prepare next iteration
        xk, fk, gk = xk_next, fk_next, gk_next

    # max iterations reached
    success = False
    if logger:
        logger.finalize(success=success, n_iter=max_iter, f_final=fk,
                        g_final=float(np.linalg.norm(gk, 2)), x_final=xk)
        logger.close()

    return {"x": xk, "f": fk, "g": gk, "n_iter": max_iter,
            "n_func_eval": n_func_eval, "n_grad_eval": n_grad_eval,
            "success": success}


def conjugate_gradient_descent(f, grad, x0, line_search, opts = None): # Polak-Ribiere apporach
    """
    Polak-Ribiere nonlinear conjugate gradient method with line search

    purpose
        minimize a smooth unconstrained function f using the nonlinear conjugate
        gradient method (Polak--Ribiere formula) with a line search. The
        implementation applies a restart (beta set to zero) when the Polak--Ribiere
        coefficient becomes negative (commonly used to maintain descent).

    inputs
        f            callable, f(x) -> scalar objective value
        grad         callable, grad(x) -> gradient vector at x  
        x0           starting point  
        line_search  callable, line_search(f, grad, xk, pk, ...) -> (alpha, f_new, n_eval)
                     a line search function that takes f, grad, current point xk and search direction pk
                     and returns a step length alpha > 0 satisfying some conditions along with
                     the new objective value f_new = f(xk + alpha pk) and the number of objective evaluations n_eval

        opts         dictionary of all the options for the optimization run, possible keys are
                        max_iter          maximum number of iterations 
                        tol               tolerance on gradient norm for termination
                        line_search_opts  dictionary of options for the line search function
                        save_flag         whether to save iteration history, default True
                        optimizer         name of the optimizer, default 'conjugate_gradient_descent'
                        line_search       name of the line search, default 'armijo_backtracking'
                        out_dir           directory to save results, default './results'
                        run_tag           tag to identify the run, default 'None'
                        alpha0           initial step length guess for line search, default 1.0

                        
    outputs
        x            best point found
        f_val        objective value at best point
        k            number of iterations performed
        n_eval       total number of objective evaluations performed
        grad_norm    norm of gradient at best point
        msg          termination message
    
    """

    opts = {} if opts is None else dict(opts)
    max_iter = int(opts.get('max_iter', 1000))          # maximum number of iterations
    tol = float(opts.get('tol', 1e-6))                  # tolerance on gradient norm for termination
    ls_opts = dict(opts.get('line_search_opts', {}))    # options for the line search function
    save_flag = bool(opts.get('save_flag', True))       # whether to save iteration history

    optimizer = str(opts.get('optimizer', 'conjugate_gradient_descent'))      # name of the optimizer
    ls_name = str(opts.get('line_search', 'armijo_backtracking'))   # name of the line search
    out_dir = str(opts.get("out_dir", "./results"))                 # directory to save results
    run_tag = str(opts.get("run_tag", "None"))                      # tag to identify the run

    meta_alpha0 = float(opts.get("alpha0", 1.0))       # initial step length guess for line search
    meta_c1 = opts.get("c1",ls_opts.get("c1", 0.001))  # armijo parameter
    meta_c2 = opts.get("c2",ls_opts.get("c2", 0.5))    # curvature parameter for strong wolfe

    logger = None
    if save_flag:
        logger = RunLogger(out_dir = out_dir,
                           optimizer = optimizer,
                           line_search = ls_name,
                           alpha0 = meta_alpha0,
                           c1 = meta_c1,
                           c2 = meta_c2,
                           max_iter = max_iter,
                           gtol = tol,
                           run_tag = run_tag)
        
    # initialization
    xk = np.asarray(x0, dtype = float)
    fk = float(f(xk))
    gk = np.asarray(grad(xk), dtype = float)

    n_func_eval = 1  # count f evaluation at initial point
    n_grad_eval = 1  # count grad evaluation at initial point

    pk = -gk  # initial steepest descent direction

    for k in range(1, max_iter + 1):

        gk_norm = float(np.linalg.norm(gk, ord = 2)) # compute gradient norm

        if gk_norm <= tol:  # check optimality
            success = True

            if logger is not None:
                # exclude initial evaluations from the counts
                logger.add_eval_counts(f_evals = n_func_eval - 1, g_evals = n_grad_eval - 1)
                logger.finalize(success = success, n_iter = k - 1, f_final = fk, g_final = float(np.linalg.norm(gk, 2)), x_final = xk)
                logger.close()

            return {"x" : xk,
                    "f" : fk,
                    "g" : gk,
                    "n_iter" : k - 1,
                    "n_func_eval" : n_func_eval,
                    "n_grad_eval" : n_grad_eval,
                    "success" : success
                    }   
        
        p_norm = float(np.linalg.norm(pk, ord = 2)) # norm of search direction

        if np.dot(gk, pk) >= 0:
            pk = -gk

        alpha, f_new, n_eval_f = line_search(f, grad, xk, pk, **ls_opts)  # call line search
        n_func_eval += int(n_eval_f) # update function evaluation count

        xk_next = xk + alpha * pk  # new point
        fk_next = float(f_new)  # new objective value
        gk_next = np.asarray(grad(xk_next), dtype = float)  # new gradient
        n_grad_eval += 1  # update gradient evaluation count

        df_abs = abs(fk - fk_next)  # absolute change in objective

        if logger:
            logger.add_eval_counts(f_evals = n_eval_f)
            logger.log_iter(int(k), float(fk_next), float(np.linalg.norm(gk_next, 2)), float(alpha), float(p_norm), float(df_abs))

        # compute beta using Polak-Ribiere formula
        yk = gk_next - gk
        eps = 1e-12
        beta = np.dot(gk_next, yk) / max(np.dot(gk, gk), eps) # beta_pr = (g_{k+1}^t (g_{k+1} - g_k)) / (g_k^t g_k)
        beta = max(beta, 0)  # ensure beta is non-negative, (natural restart)


        # # update search direction
        pk = -gk_next + beta * pk

        # Restart condition with v = 0.2
        cos_angle = abs(np.dot(gk_next, pk)) / (np.linalg.norm(gk_next) * np.linalg.norm(pk) + 1e-12)
        if cos_angle > 0.2:
            pk = -gk_next

        # accept the new point
        xk = xk_next    
        fk = fk_next
        gk = gk_next

    # iteration limit reached without convergence
    success = False
    
    if logger: 
        logger.finalize(success = success, n_iter = max_iter, f_final = fk, g_final = float(np.linalg.norm(gk, 2)), x_final = xk)
        logger.close()

    return {"x" : xk,
            "f" : fk,
            "g" : gk,
            "n_iter" : max_iter,
            "n_func_eval" : n_func_eval,
            "n_grad_eval" : n_grad_eval,
            "success" : success
    }


def quasi_newton_bfgs(f, grad, x0, line_search, opts = None):
    """
    quasi-newton method with BFGS update and line search
    """
    opts = {} if opts is None else dict(opts)
    max_iter = int(opts.get('max_iter', 1000))          # maximum number of iterations
    tol = float(opts.get('tol', 1e-6))                  # tolerance on gradient norm for termination
    ls_opts = dict(opts.get('line_search_opts', {}))    # options for the line search function
    save_flag = bool(opts.get('save_flag', True))       # whether to save iteration history

    optimizer = str(opts.get('optimizer', 'quasi_newton_bfgs'))      # name of the optimizer
    ls_name = str(opts.get('line_search', 'strong_wolfe'))   # name of the line search
    out_dir = str(opts.get("out_dir", "./results"))                 # directory to save results
    run_tag = str(opts.get("run_tag", "None"))                      # tag to identify the run

    meta_alpha0 = float(opts.get("alpha0", 1.0))       # initial step length guess for line search
    meta_c1 = opts.get("c1",ls_opts.get("c1", 0.001))  # armijo parameter
    meta_c2 = opts.get("c2",ls_opts.get("c2", 0.5))    # curvature parameter for strong wolfe

    logger = None
    if save_flag:
        logger = RunLogger(out_dir = out_dir,
                           optimizer = optimizer,
                           line_search = ls_name,
                           alpha0 = meta_alpha0,
                           c1 = meta_c1,
                           c2 = meta_c2,
                           max_iter = max_iter,
                           gtol = tol,
                           run_tag = run_tag)
        
    # initialization
    xk = np.asarray(x0, dtype = float)
    fk = float(f(xk))
    gk = np.asarray(grad(xk), dtype = float)

    n_func_eval = 1  # count f evaluation at initial point
    n_grad_eval = 1  # count grad evaluation at initial point

    n_dim = xk.size
    Hk = np.eye(n_dim)  # initial inverse hessian approximation (identity)

    for k in range(1, max_iter + 1):

        gk_norm = float(np.linalg.norm(gk, ord = 2)) # compute gradient norm

        if gk_norm <= tol:  # check optimality
            success = True

            if logger is not None:
                # exclude initial evaluations from the counts
                logger.add_eval_counts(f_evals = n_func_eval - 1, g_evals = n_grad_eval - 1)
                logger.finalize(success = success, n_iter = k - 1, f_final = fk, g_final = float(np.linalg.norm(gk, 2)), x_final = xk)
                logger.close()

            return {"x" : xk,
                    "f" : fk,
                    "g" : gk,
                    "n_iter" : k - 1,
                    "n_func_eval" : n_func_eval,
                    "n_grad_eval" : n_grad_eval,
                    "success" : success
                    }
        
        # compute search direction
        pk = -np.dot(Hk, gk)  
        p_norm = float(np.linalg.norm(pk, ord = 2)) 

        # call line search
        alpha, f_new, n_eval_f = line_search(f, grad, xk, pk, **ls_opts)  
        n_func_eval += int(n_eval_f) 

        # update step
        xk_next = xk + alpha * pk  
        fk_next = float(f_new)  
        gk_next = np.asarray(grad(xk_next), dtype = float)  
        n_grad_eval += 1  

        df_abs = abs(fk - fk_next)  # absolute change in objective
        if logger:
            logger.add_eval_counts(f_evals = n_eval_f)
            logger.log_iter(int(k), float(fk_next), float(np.linalg.norm(gk_next, 2)), float(alpha), float(p_norm), float(df_abs))

        # BFGS update
        sk = xk_next - xk  # step
        yk = gk_next - gk  # gradient difference
        syk = np.dot(sk, yk)  # sk^T yk

        if syk > 0.0:  # update Hk only if sk^T yk is sufficiently positive to maintain positive definiteness
            rho_k = 1.0 / syk
            I = np.eye(n_dim)
            syT = np.outer(sk, yk)
            ysT = np.outer(yk, sk)
            ssT = np.outer(sk, sk)
            v = I - rho_k * syT
            Hk = np.dot(v, np.dot(Hk, v.T)) + rho_k * ssT  # BFGS formula

        # accept the new point
        xk = xk_next
        fk = fk_next
        gk = gk_next

    # iteration limit reached without convergence
    success = False
    if logger: 
        logger.finalize(success = success, n_iter = max_iter, f_final = fk, g_final = float(np.linalg.norm(gk, 2)), x_final = xk)
        logger.close()

    return {"x" : xk,
            "f" : fk,
            "g" : gk,
            "n_iter" : max_iter,
            "n_func_eval" : n_func_eval,
            "n_grad_eval" : n_grad_eval,
            "success" : success
    }


# ----------------------------------------------------------------
# fast feedforward neural networkjective
# ----------------------------------------------------------------

def make_ffnn_objective(X, y, layer_sizes, l2 = 0.0):
    """
    create closures f(w) and grad(w) for a general ffnn

    inputs
        X            array (n, d_in), features
        y            array (n,) or (n, 1), targets in {1/6, ..., 5/6}
        layer_sizes  list like [d_in, h1, ..., d_out] with d_out = 1
        l2           nonnegative l2 regularization on weights (biases excluded)

    returns
        f            callable, f(w) -> scalar loss
        grad         callable, grad(w) -> flat gradient vector
    """

    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float).reshape(-1, 1)

    n, d_in = X.shape
    assert layer_sizes[0] == d_in, "input dimension must match layer_sizes[0]"
    assert layer_sizes[-1] == 1, "output dimension must be 1 for this setup"

    # precompute parameter slices 
    # for each layer l: W in R^{d_{l-1} x d_l}, b in R^{d_l}
    shapes = []
    sizes = []
    for l in range(1, len(layer_sizes)):
        d_prev = layer_sizes[l - 1]
        d_curr = layer_sizes[l]
        w_count = d_prev * d_curr
        b_count = d_curr
        shapes.append(((d_prev, d_curr), (d_curr,)))
        sizes.append((w_count, b_count))

    # compute flat index ranges
    idx = 0
    slices = []
    for (w_count, b_count) in sizes:
        sW = slice(idx, idx + w_count)
        idx += w_count
        sb = slice(idx, idx + b_count)
        idx += b_count
        slices.append((sW, sb))
    p_expected = idx

    # cached activations for backprop; created locally inside f/grad
    def _sigmoid(u):
        return 1.0 / (1.0 + np.exp(-u))

    def f(w):
        w = np.asarray(w, dtype = float)
        if w.size != p_expected:
            raise ValueError("size of w does not match architecture")

        # unpack and forward pass
        a_list = [X]   # a_0
        params = []
        a = X
        for (sW, sb), (shapeW, shapeb) in zip(slices, shapes):
            W = w[sW].reshape(shapeW)
            b = w[sb].reshape(shapeb)
            params.append((W, b))
            z = a @ W + b
            a = _sigmoid(z)
            a_list.append(a)

        yhat = a_list[-1]                         # (n, 1)
        residual = yhat - y
        loss = 0.5 * np.sum(residual * residual)

        if l2 > 0.0:
            reg = 0.0
            for (W, b) in params:
                reg += np.sum(W * W)
            loss += 0.5 * l2 * reg

        return float(loss)

    def grad(w):
        w = np.asarray(w, dtype = float)
        if w.size != p_expected:
            raise ValueError("size of w does not match architecture")

        # unpack and forward pass (store activations for backprop)
        a_list = [X]
        z_list = []
        params = []
        a = X
        for (sW, sb), (shapeW, shapeb) in zip(slices, shapes):
            W = w[sW].reshape(shapeW)
            b = w[sb].reshape(shapeb)
            params.append((W, b))
            z = a @ W + b
            a = _sigmoid(z)
            z_list.append(z)
            a_list.append(a)

        yhat = a_list[-1]

        # initial delta at output: (yhat - y) * sigma'(z_L)
        # derivative of sigmoid via activation: a * (1 - a)
        delta = (yhat - y) * (yhat * (1.0 - yhat))   # (n, 1)

        # backprop
        dWs = [None] * len(params)
        dbs = [None] * len(params)

        for l in reversed(range(len(params))):
            a_prev = a_list[l]                       # (n, d_l)
            W, b = params[l]                         # W: (d_l, d_{l+1}), b: (d_{l+1},)

            dW = a_prev.T @ delta                    # (d_l, d_{l+1})
            db = np.sum(delta, axis = 0)             # (d_{l+1},)
            if l2 > 0.0:
                dW += l2 * W

            dWs[l] = dW
            dbs[l] = db

            if l > 0:
                a_prev_act = a_list[l]               # activation at layer l
                delta = (delta @ W.T) * (a_prev_act * (1.0 - a_prev_act))

        # pack grads into a flat vector following the same order
        g = np.empty(p_expected, dtype = float)
        for (sW, sb), dW, db in zip(slices, dWs, dbs):
            g[sW] = dW.ravel()
            g[sb] = db.ravel()

        return g

    return f, grad


def ffnn_init_params(layer_sizes, random_state = None):
    """
    xavier/glorot uniform initializer for sigmoid networks

    inputs
        layer_sizes   [d_in, h1, ..., d_out]
        random_state  seed or numpy random generator or None

    returns
        w0            flat parameter vector
    """
    if random_state is None:
        random_state = np.random.default_rng(42)

    blocks = []
    for l in range(1, len(layer_sizes)):
        d_prev = layer_sizes[l - 1]
        d_curr = layer_sizes[l]
        limit = np.sqrt(6.0 / (d_prev + d_curr))
        W = random_state.uniform(-limit, limit, size = (d_prev, d_curr))
        b = np.zeros((d_curr,), dtype = float)
        blocks.append(W.ravel())
        blocks.append(b)
    return np.concatenate(blocks, axis = 0)




# ----------------------------------------------------------------
# Miscellaneous functions
# ----------------------------------------------------------------

# stratified train-test split for dataframes
def stratified_split_df(df, target_col, test_size = 0.3, random_state = 42):
    """
    stratified split by discrete target values

    inputs
        df           dataframe to split
        target_col   column with class labels (here: original 1..5 happiness)
        test_size    fraction in test
        random_state seed for reproducibility

    outputs
        df_train, df_test
    """
    rng = np.random.default_rng(random_state)

    df_train_parts = []
    df_test_parts  = []

    for cls, grp in df.groupby(target_col):
        n = len(grp)
        n_test = int(round(test_size * n))
        n_test = max(1, min(n_test, n - 1))  # at least 1 in each side when possible

        idx = np.arange(n)
        rng.shuffle(idx)

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        df_test_parts.append(grp.iloc[test_idx])
        df_train_parts.append(grp.iloc[train_idx])

    df_train = pd.concat(df_train_parts, axis = 0).sample(frac = 1.0, random_state = random_state).reset_index(drop = True)
    df_test  = pd.concat(df_test_parts,  axis = 0).sample(frac = 1.0, random_state = random_state).reset_index(drop = True)

    return df_train, df_test