import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse.linalg import cg
import os

# ============================================================================
# PROBLEM DEFINITION
# ============================================================================

# Fixed endpoints
z0 = np.array([4.0, 5.0])
z9 = np.array([26.0, 5.0])

# Cost coefficients
beta = 1.0
alpha_values = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]  # Task 4: various alpha values

# Polygon vertices (counter-clockwise order)
P0_vertices = np.array([[2, 1], [6.2, 2.8], [7.7, 7.9], [2, 8], [0.4, 4.5]])
P1_vertices = np.array([[9.9, 3.6], [11.6, 2.7], [13.2, 5.7], [10.4, 6.3]])
P2_vertices = np.array([[15.5, 5.7], [16.3, 3.4], [18.2, 4.3], [18.2, 7.5], [16.1, 7.6]])
P3_vertices = np.array([[21.6, 0.8], [23.5, 9.5], [21.7, 7.7]])
P4_vertices = np.array([[23.7, 5.5], [24.9, 3.7], [27.6, 5.6], [25, 6.7]])

polygons = [P0_vertices, P1_vertices, P2_vertices, P3_vertices, P4_vertices]

# Task 5 settings: Convert polygon to line segment
TASK5_ENABLED = True  # Set to True to run Task 5
TASK5_POLYGON_IDX = 2  # Which polygon to convert (0-4)
TASK5_EDGE_IDX = 1     # Which edge of that polygon (0 to num_vertices-1)
TASK5_ALPHA = 1.5      # Alpha value for Task 5

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def vertices_to_halfspaces(vertices):
    """
    Convert counter-clockwise vertices to inequality constraints Ax <= b
    For polygon P = conv{(a1,b1),...,(ap,bp)}, compute:
    (bk+1 - bk)*x + (ak - ak+1)*y <= ak*bk+1 - bk*ak+1
    """
    n = len(vertices)
    A_ineq = []
    b_ineq = []
    
    for k in range(n):
        a_k, b_k = vertices[k]
        a_kp1, b_kp1 = vertices[(k + 1) % n]
        
        # Inequality: (b_{k+1} - b_k)*x + (a_k - a_{k+1})*y <= a_k*b_{k+1} - b_k*a_{k+1}
        A_row = [b_kp1 - b_k, a_k - a_kp1]
        b_val = a_k * b_kp1 - b_k * a_kp1
        
        A_ineq.append(A_row)
        b_ineq.append(b_val)
    
    return np.array(A_ineq), np.array(b_ineq)


def get_edge_as_equality(vertices, edge_idx):
    """
    Convert one edge of polygon to equality constraint
    Returns A_eq, b_eq representing the line through the edge
    """
    n = len(vertices)
    p1 = vertices[edge_idx]
    p2 = vertices[(edge_idx + 1) % n]
    
    # Line through p1 and p2: (y2-y1)*x - (x2-x1)*y = y2*x1 - x2*y1
    A_eq = np.array([[p2[1] - p1[1], -(p2[0] - p1[0])]])
    b_eq = np.array([p2[1] * p1[0] - p2[0] * p1[1]])
    
    return A_eq, b_eq, p1, p2


# ============================================================================
# TASK 2: BUILD QP MATRICES
# ============================================================================

def build_transmission_qp(polygons, z0, z9, beta, alpha):
    """
    Build QP matrices for transmission line problem.
    
    Variables ordered as: [z1_x, z1_y, z2_x, z2_y, ..., z8_x, z8_y]
    
    Returns: G, c, A, b, Ae, be
    """
    # Number of variables
    n_vars = 16  # 8 points × 2 coordinates
    
    # Initialize matrices
    G = np.zeros((n_vars, n_vars))
    c = np.zeros(n_vars)
    
    # Build objective: sum of squared distances
    # Beta terms: ||z0-z1||^2, ||z2-z3||^2, ||z4-z5||^2, ||z6-z7||^2, ||z8-z9||^2
    # Alpha terms: ||z1-z2||^2, ||z3-z4||^2, ||z5-z6||^2, ||z7-z8||^2
    
    for k in range(5):  # k = 0, 1, 2, 3, 4
        if k == 0:
            # ||z0 - z1||^2 with coefficient beta
            # z1 is vars[0:2]
            G[0:2, 0:2] += beta * np.eye(2)
            c[0:2] += -beta * z0
        elif k == 4:
            # ||z8 - z9||^2 with coefficient beta
            # z8 is vars[14:16]
            G[14:16, 14:16] += beta * np.eye(2)
            c[14:16] += -beta * z9
        else:
            # ||z_{2k} - z_{2k+1}||^2 with coefficient beta
            idx1 = 2 * (2 * k) - 2  # z_{2k} position
            idx2 = 2 * (2 * k + 1) - 2  # z_{2k+1} position
            G[idx1:idx1+2, idx1:idx1+2] += beta * np.eye(2)
            G[idx2:idx2+2, idx2:idx2+2] += beta * np.eye(2)
            G[idx1:idx1+2, idx2:idx2+2] += -beta * np.eye(2)
            G[idx2:idx2+2, idx1:idx1+2] += -beta * np.eye(2)
    
    for k in range(1, 5):  # k = 1, 2, 3, 4
        # ||z_{2k-1} - z_{2k}||^2 with coefficient alpha
        idx1 = 2 * (2 * k - 1) - 2
        idx2 = 2 * (2 * k) - 2
        G[idx1:idx1+2, idx1:idx1+2] += alpha * np.eye(2)
        G[idx2:idx2+2, idx2:idx2+2] += alpha * np.eye(2)
        G[idx1:idx1+2, idx2:idx2+2] += -alpha * np.eye(2)
        G[idx2:idx2+2, idx1:idx1+2] += -alpha * np.eye(2)
    
    # Build constraint matrices
    A_list = []
    b_list = []
    
    for k in range(1, 9):  # z1 through z8
        poly_idx = k // 2
        if isinstance(polygons[poly_idx], dict) and 'type' in polygons[poly_idx]:
            # Special case: line segment
            if polygons[poly_idx]['type'] == 'line_segment':
                # Handle as inequality + equality constraints
                p1, p2 = polygons[poly_idx]['p1'], polygons[poly_idx]['p2']
                
                # Inequality: bounding box constraints
                x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
                y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
                

                var_idx = 2 * (k - 1)
                row = np.zeros(n_vars)
                row[var_idx] = -1
                A_list.append(row)
                b_list.append(-x_min)
                
                row = np.zeros(n_vars)
                row[var_idx] = 1
                A_list.append(row)
                b_list.append(x_max)
                
                row = np.zeros(n_vars)
                row[var_idx + 1] = -1
                A_list.append(row)
                b_list.append(-y_min)
                
                row = np.zeros(n_vars)
                row[var_idx + 1] = 1
                A_list.append(row)
                b_list.append(y_max)
        else:

            A_poly, b_poly = vertices_to_halfspaces(polygons[poly_idx])
            var_idx = 2 * (k - 1) 
            
            for i in range(len(A_poly)):
                row = np.zeros(n_vars)
                row[var_idx:var_idx+2] = A_poly[i]
                A_list.append(row)
                b_list.append(b_poly[i])
    
    A = np.array(A_list) if A_list else np.zeros((0, n_vars))
    b = np.array(b_list) if b_list else np.zeros(0)
    
    # Equality constraints (for line segments)
    Ae_list = []
    be_list = []
    
    for k in range(1, 9):
        poly_idx = k // 2
        if isinstance(polygons[poly_idx], dict) and polygons[poly_idx]['type'] == 'line_segment':
            A_eq_poly = polygons[poly_idx]['A_eq']
            b_eq_poly = polygons[poly_idx]['b_eq']
            var_idx = 2 * (k - 1)
            
            row = np.zeros(n_vars)
            row[var_idx:var_idx+2] = A_eq_poly[0]
            Ae_list.append(row)
            be_list.append(b_eq_poly[0])
    
    Ae = np.array(Ae_list) if Ae_list else np.zeros((0, n_vars))
    be = np.array(be_list) if be_list else np.zeros(0)
    
    return G, c, A, b, Ae, be


# ============================================================================
# TASK 3: QP SOLVERS
# ============================================================================

def solve_qp(G, c, A, b, Ae, be, x0=None):
    """
    Solve general QP: min 0.5*x'Gx + c'x
                      s.t. Ax <= b
                           Ae*x = be
    
    Handles 4 cases:
    (a) Linear program (G = 0)
    (b) Unconstrained QP
    (c) Equality-constrained QP
    (d) Inequality-constrained QP (Active Set Method)
    """
    n = len(c)
    
    if x0 is None:
        x0 = np.zeros(n)
    
    # Check if G is zero (linear program)
    if np.allclose(G, 0):
        return solve_lp(c, A, b, Ae, be)
    
    # Check if unconstrained
    if (A.size == 0 or len(A) == 0) and (Ae.size == 0 or len(Ae) == 0):
        return solve_unconstrained_qp(G, c)
    
    # Check if only equality constraints
    if A.size == 0 or len(A) == 0:
        return solve_equality_qp(G, c, Ae, be)
    
    # General case: inequality constraints (with possible equality constraints)
    return solve_active_set_qp(G, c, A, b, Ae, be, x0)


def solve_lp(c, A, b, Ae, be):
    """Solve LP using scipy.optimize.linprog"""
    # linprog minimizes c'x subject to A_ub @ x <= b_ub and A_eq @ x == b_eq
    A_ub = A if A.size > 0 else None
    b_ub = b if b.size > 0 else None
    A_eq = Ae if Ae.size > 0 else None
    b_eq = be if be.size > 0 else None
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    
    if result.success:
        return result.x
    else:
        raise ValueError("LP solver failed")


def solve_unconstrained_qp(G, c):
    """Solve unconstrained QP: min 0.5*x'Gx + c'x using Newton step"""
    # Optimal: G*x + c = 0 => x = -G^{-1} * c
    try:
        x = np.linalg.solve(G, -c)
        return x
    except np.linalg.LinAlgError:
        x = -np.linalg.pinv(G) @ c
        return x


def solve_equality_qp(G, c, Ae, be):
    """
    Solve equality-constrained QP using CG method (null space approach)
    min 0.5*x'Gx + c'x subject to Ae*x = be
    """
    # KKT system: [G  Ae'] [x  ] = [-c ]
    #             [Ae  0 ] [lam]   [be ]
    
    n = G.shape[0]
    m = Ae.shape[0] if Ae.size > 0 else 0
    
    if m == 0:
        return solve_unconstrained_qp(G, c)
    
    # Build KKT matrix
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = G
    KKT[:n, n:] = Ae.T
    KKT[n:, :n] = Ae
    
    rhs = np.concatenate([-c, be])
    
    try:
        sol = np.linalg.solve(KKT, rhs)
        return sol[:n]
    except np.linalg.LinAlgError:
        # Use least squares
        sol, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
        return sol[:n]


def solve_active_set_qp(G, c, A, b, Ae, be, x0, max_iter=1000, tol=1e-10):
    """
    Solve QP with inequality constraints using Active Set Method
    """
    n = len(c)
    m = len(b) if b.size > 0 else 0
    me = len(be) if be.size > 0 else 0
    
    # Find initial feasible point
    x = find_feasible_point(A, b, Ae, be, x0)
    
    # Initialize active set 
    active_set = set(range(m, m + me))  # Equality constraints always active
    
    # Check which inequalities are active at x0
    if m > 0:
        slack = b - A @ x
        for i in range(m):
            if abs(slack[i]) < tol:
                active_set.add(i)
    
    for iteration in range(max_iter):
        # Build active constraint matrix
        active_indices = list(active_set)
        if len(active_indices) > 0:
            A_active = []
            b_active = []
            for idx in active_indices:
                if idx < m:
                    A_active.append(A[idx])
                    b_active.append(b[idx])
                else:
                    A_active.append(Ae[idx - m])
                    b_active.append(be[idx - m])
            A_active = np.array(A_active)
            b_active = np.array(b_active)
        else:
            A_active = np.zeros((0, n))
            b_active = np.zeros(0)
        
        # Solve equality-constrained QP for search direction
        # min 0.5*p'Gp + (Gx+c)'p subject to A_active * p = 0
        if len(active_indices) > 0:
            p = solve_equality_qp(G, G @ x + c, A_active, np.zeros(len(active_indices)))
        else:
            p = solve_unconstrained_qp(G, G @ x + c)
        
        # Check if p is zero (KKT point)
        if np.linalg.norm(p) < tol:
            # Compute Lagrange multipliers
            if len(active_indices) > 0:
                lam = compute_lagrange_multipliers(G, c, x, A_active)
                
                # Check if all multipliers for inequalities are non-negative
                min_lam_idx = -1
                min_lam = 0
                for i, idx in enumerate(active_indices):
                    if idx < m:  # Inequality constraint
                        if lam[i] < min_lam:
                            min_lam = lam[i]
                            min_lam_idx = idx
                
                if min_lam < -tol:
                    # Remove constraint with most negative multiplier
                    active_set.remove(min_lam_idx)
                    continue
            
            # KKT conditions satisfied
            break
        
        # Compute step size (maximum step before hitting constraint)
        alpha = 1.0
        blocking_constraint = -1
        
        if m > 0:
            Ap = A @ p
            for i in range(m):
                if i not in active_set and Ap[i] > tol:
                    alpha_i = (b[i] - A[i] @ x) / Ap[i]
                    if alpha_i < alpha:
                        alpha = alpha_i
                        blocking_constraint = i
        
        # Update x
        x = x + alpha * p
        
        # Add blocking constraint to active set
        if blocking_constraint >= 0:
            active_set.add(blocking_constraint)
    
    return x


def find_feasible_point(A, b, Ae, be, x0):
    """Find a feasible starting point for QP"""
    n = len(x0)
    
    # Check if x0 is feasible
    feasible = True
    if A.size > 0 and len(A) > 0:
        if np.any(A @ x0 > b + 1e-6):
            feasible = False
    if Ae.size > 0 and len(Ae) > 0:
        if not np.allclose(Ae @ x0, be, atol=1e-6):
            feasible = False
    
    if feasible:
        return x0.copy()
    
    # Solve phase 1 problem to find feasible point
    # min s subject to Ax <= b + s, Ae*x = be, s >= 0
    m = len(b) if b.size > 0 else 0
    me = len(be) if be.size > 0 else 0
    
    # Variables: [x, s]
    c_phase1 = np.zeros(n + 1)
    c_phase1[-1] = 1  # Minimize s
    
    # Inequality constraints: Ax - s <= b
    if m > 0:
        A_phase1 = np.hstack([A, -np.ones((m, 1))])
        b_phase1 = b
    else:
        A_phase1 = np.zeros((0, n + 1))
        b_phase1 = np.zeros(0)
    
    # s >= 0 => -s <= 0
    s_constraint = np.zeros((1, n + 1))
    s_constraint[0, -1] = -1
    A_phase1 = np.vstack([A_phase1, s_constraint]) if A_phase1.size > 0 else s_constraint
    b_phase1 = np.concatenate([b_phase1, [0]]) if b_phase1.size > 0 else np.array([0])
    
    # Equality constraints: Ae*x = be (s doesn't appear)
    if me > 0:
        Ae_phase1 = np.hstack([Ae, np.zeros((me, 1))])
        be_phase1 = be
    else:
        Ae_phase1 = np.zeros((0, n + 1))
        be_phase1 = np.zeros(0)
    
    result = linprog(c_phase1, A_ub=A_phase1, b_ub=b_phase1, 
                     A_eq=Ae_phase1 if Ae_phase1.size > 0 else None,
                     b_eq=be_phase1 if be_phase1.size > 0 else None,
                     method='highs')
    
    if result.success and result.fun < 1e-6:
        return result.x[:n]
    else:
        # Return x0 and hope for the best
        return x0.copy()


def compute_lagrange_multipliers(G, c, x, A_active):
    """Compute Lagrange multipliers for active constraints"""
    # At optimum: G*x + c = A_active' * lambda
    # => lambda = (A_active * A_active')^{-1} * A_active * (G*x + c)
    grad = G @ x + c
    
    try:
        AAt = A_active @ A_active.T
        lam = np.linalg.solve(AAt, A_active @ grad)
        return lam
    except np.linalg.LinAlgError:
        # Singular, use pseudo-inverse
        lam = np.linalg.pinv(A_active.T) @ grad
        return lam


# ============================================================================
# PLOTTING AND OUTPUT
# ============================================================================

def plot_solution(polygons, z0, z9, z_opt, alpha, beta, filename):
    """Plot polygons and optimal path"""
    plt.figure(figsize=(14, 8))
    
    # Plot polygons
    for i, poly in enumerate(polygons):
        if isinstance(poly, dict) and poly['type'] == 'line_segment':
            # Plot line segment
            p1, p2 = poly['p1'], poly['p2']
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo', markersize=6)
            plt.text(np.mean([p1[0], p2[0]]), np.mean([p1[1], p2[1]]) + 0.5, 
                    f'$P_{i}$ (segment)', color='blue', fontsize=10, ha='center')
        else:
            # Plot polygon
            poly_closed = np.vstack([poly, poly[0]])
            plt.plot(poly_closed[:, 0], poly_closed[:, 1], 'b-', linewidth=2)
            centroid = np.mean(poly, axis=0)
            plt.text(centroid[0], centroid[1], f'$P_{i}$', color='blue', 
                    fontsize=12, ha='center', weight='bold')
    
    # Plot optimal path
    all_points = [z0] + [z_opt[2*i:2*i+2] for i in range(8)] + [z9]
    path = np.array(all_points)
    plt.plot(path[:, 0], path[:, 1], 'o-', color='orange', linewidth=3, 
             markersize=8, label='Optimal Path')
    
    # Label points
    plt.plot(z0[0], z0[1], 'ko', markersize=10)
    plt.text(z0[0] - 0.1, z0[1]+ 0.5, '$z_0$', fontsize=12, ha='right')
    
    for i in range(8):
        zi = z_opt[2*i:2*i+2]
        plt.plot(zi[0], zi[1], 'ro', markersize=8)
        # Shift label up and to the right
        plt.text(zi[0] - 0.5, zi[1] - 0.5, f'$z_{i+1}$', fontsize=12, ha='left', va='bottom')
    
    plt.plot(z9[0], z9[1], 'ko', markersize=10)
    plt.text(z9[0] - 0.1, z9[1]+ 0.5, '$z_9$', fontsize=12, ha='left')
    
    plt.xlabel('x', fontsize=18)
    plt.xticks(fontsize=14)
    plt.ylabel('y', fontsize=18)
    plt.yticks(fontsize=14)
    # plt.title(f'Optimal Transmission Line Layout (α={alpha}, β={beta})', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()



def compute_objective(z_opt, z0, z9, beta, alpha):
    """Compute the objective function value"""
    obj = 0.0
    
    # Beta terms
    for k in range(5):
        if k == 0:
            z1 = z_opt[0:2]
            obj += beta / 2 * np.sum((z0 - z1) ** 2)
        elif k == 4:
            z8 = z_opt[14:16]
            obj += beta / 2 * np.sum((z8 - z9) ** 2)
        else:
            z2k = z_opt[2*(2*k)-2:2*(2*k)]
            z2k1 = z_opt[2*(2*k+1)-2:2*(2*k+1)]
            obj += beta / 2 * np.sum((z2k - z2k1) ** 2)
    
    # Alpha terms
    for k in range(1, 5):
        z2k_1 = z_opt[2*(2*k-1)-2:2*(2*k-1)]
        z2k = z_opt[2*(2*k)-2:2*(2*k)]
        obj += alpha / 2 * np.sum((z2k_1 - z2k) ** 2)
    
    return obj


def save_results(results, filename):
    """Save results to text file"""
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRANSMISSION LINE OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Alpha: {result['alpha']:.2f}, Beta: {result['beta']:.2f}\n")
            f.write(f"Objective Value: {result['objective']:.6f}\n")
            f.write(f"Alpha/Beta Ratio: {result['alpha']/result['beta']:.2f}\n")
            f.write("\nOptimal Points:\n")
            for i in range(8):
                zi = result['z_opt'][2*i:2*i+2]
                f.write(f"  z{i+1} = ({zi[0]:.4f}, {zi[1]:.4f})\n")
            f.write("\n" + "-" * 80 + "\n\n")


# ============================================================================
# TASK 4: SOLVE FOR VARIOUS ALPHA VALUES
# ============================================================================

def run_task4():
    """Run Task 4: Solve for various alpha values"""
    print("\n" + "=" * 80)
    print("TASK 4: Solving transmission line problem for various alpha values")
    print("=" * 80)
    
    results = []
    
    for alpha in alpha_values:
        print(f"\nSolving for alpha = {alpha}, beta = {beta}...")
        
        # Build QP
        G, c, A, b, Ae, be = build_transmission_qp(polygons, z0, z9, beta, alpha)
        
        # Initial guess (centroid of each polygon)
        x0 = []
        for k in range(1, 9):
            poly_idx = k // 2
            centroid = np.mean(polygons[poly_idx], axis=0)
            x0.extend(centroid)
        x0 = np.array(x0)
        
        # Solve QP
        z_opt = solve_qp(G, c, A, b, Ae, be, x0)
        
        # Compute objective
        obj = compute_objective(z_opt, z0, z9, beta, alpha)
        
        print(f"  Objective value: {obj:.6f}")
        
        # Save results
        result = {
            'alpha': alpha,
            'beta': beta,
            'z_opt': z_opt,
            'objective': obj
        }
        results.append(result)
        
        # Plot
        plot_filename = f"task4_alpha_{alpha:.1f}.png"
        plot_solution(polygons, z0, z9, z_opt, alpha, beta, plot_filename)
        print(f"  Plot saved to {plot_filename}")
    
    # Save all results
    save_results(results, "task4_results.txt")
    print("\nAll results saved to task4_results.txt")
    
    return results


# ============================================================================
# TASK 5: LINE SEGMENT CONSTRAINT
# ============================================================================

def run_task5():
    """Run Task 5: Convert one polygon to line segment"""
    print("\n" + "=" * 80)
    print("TASK 5: Solving with line segment constraint")
    print("=" * 80)
    
    # Modify polygons
    polygons_task5 = polygons.copy()
    vertices = polygons_task5[TASK5_POLYGON_IDX]
    
    # Get edge as line segment
    A_eq, b_eq, p1, p2 = get_edge_as_equality(vertices, TASK5_EDGE_IDX)
    
    print(f"\nConverting polygon P{TASK5_POLYGON_IDX}, edge {TASK5_EDGE_IDX} to line segment")
    print(f"  Segment endpoints: {p1} to {p2}")
    
    # Replace polygon with line segment info
    polygons_task5[TASK5_POLYGON_IDX] = {
        'type': 'line_segment',
        'p1': p1,
        'p2': p2,
        'A_eq': A_eq,
        'b_eq': b_eq
    }
    
    # Build and solve QP
    print(f"Solving for alpha = {TASK5_ALPHA}, beta = {beta}...")
    G, c, A, b, Ae, be = build_transmission_qp(polygons_task5, z0, z9, beta, TASK5_ALPHA)
    
    # Initial guess
    x0 = []
    for k in range(1, 9):
        poly_idx = k // 2
        if isinstance(polygons_task5[poly_idx], dict):
            # Use midpoint of line segment
            p1, p2 = polygons_task5[poly_idx]['p1'], polygons_task5[poly_idx]['p2']
            centroid = (p1 + p2) / 2
        else:
            centroid = np.mean(polygons_task5[poly_idx], axis=0)
        x0.extend(centroid)
    x0 = np.array(x0)
    
    # Solve
    z_opt = solve_qp(G, c, A, b, Ae, be, x0)
    
    # Compute objective
    obj = compute_objective(z_opt, z0, z9, beta, TASK5_ALPHA)
    
    print(f"  Objective value: {obj:.6f}")
    
    # Verify points on line segment
    k_segment = [k for k in range(1, 9) if k // 2 == TASK5_POLYGON_IDX]
    print(f"\nVerifying points on line segment:")
    for k in k_segment:
        zi = z_opt[2*(k-1):2*(k-1)+2]
        dist_to_line = abs(A_eq[0, 0] * zi[0] + A_eq[0, 1] * zi[1] - b_eq[0]) / np.linalg.norm(A_eq[0])
        print(f"  z{k} = ({zi[0]:.4f}, {zi[1]:.4f}), distance to line: {dist_to_line:.6e}")
    
    # Save results
    result = [{
        'alpha': TASK5_ALPHA,
        'beta': beta,
        'z_opt': z_opt,
        'objective': obj
    }]
    save_results(result, "task5_results.txt")
    
    # Plot
    plot_solution(polygons_task5, z0, z9, z_opt, TASK5_ALPHA, beta, "task5_solution.png")
    print("\nPlot saved to task5_solution.png")
    print("Results saved to task5_results.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Create results folder in same directory as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)
    
    # Run Task 4
    task4_results = run_task4()
    
    # Run Task 5 if enabled
    if TASK5_ENABLED:
        run_task5()
    
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)
