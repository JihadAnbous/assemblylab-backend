import jax
import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np
import sympy as sp

# Levenberg-Marquardt method (LM method)

# Points
# Create an index map

# Variables array


# Residuals
def calculate_residuals(beta):
    x, y, z = beta
    r1 = x**2 + y - 5
    r2 = z * np.sin(x) - 1
    return np.array([r1, r2])


# Jacobian [m x n] matrix
# m : number of residuals | rows
# n : number of variables | columns
def calculate_jacobian(beta):
    x, y, z = beta
    # Row 1 derivatives
    dr1_dx = 2 * x
    dr1_dy = 1
    dr1_dz = 0

    # Row 2 derivatives
    dr2_dx = z * np.cos(x)
    dr2_dy = 0
    dr2_dz = np.sin(x)

    return np.array([[dr1_dx, dr1_dy, dr1_dz], [dr2_dx, dr2_dy, dr2_dz]])


# Initial setup
beta = np.array([1.0, 1.0, 1.0])  # Initial guess [x, y, z]
lam = 0.1  # Damping factor
tol = 1e-8  # Convergence tolerance
max_iter = 50  # Maximum iterations

# Solver
# Loop through the maximum iterations
for i in range(max_iter):
    r = calculate_residuals(beta)
    J = calculate_jacobian(beta)

    # Sum of Squares - Sum of residual errors
    S_old = np.sum(r**2)

    # (J^T J + lambda * I) * delta = J^T * r
    jtj = J.T @ J
    gradient = J.T @ r

    # The LM update (Damping)
    H_damped = jtj + lam * np.eye(len(beta))

    try:
        # H_damped * delta_beta = gradient > np.linalg.solve() finds delta_beta
        delta_beta = np.linalg.solve(H_damped, gradient)
    except np.linalg.LinAlgError:
        # If the matrix is still singular/non-invertible, increase damping
        lam = lam * 10
        continue  # Skip the rest of this iteration

    # Test the new position
    beta_new = beta - delta_beta
    r_new = calculate_residuals(beta_new)
    S_new = np.sum(r_new**2)

    if S_new < S_old:
        # Success!
        print(f"Iter {i}: Error {S_new:.6f}, lam {lam:.4f}")
        beta = beta_new
        lam = lam / 10  # Get faster
        if np.linalg.norm(delta_beta) < tol:
            print("Converged!")
            break
    else:
        # Failure!
        lam = lam * 10  # Get more stable

print(f"\nFinal Parameters:\nx: {beta[0]:.4f}\ny: {beta[1]:.4f}\nz: {beta[2]:.4f}")


# Automating it with sympy


def setup_system(equations_list, variables_list):
    """
    equations_list: List of strings like ["x**2 + y - 5", "z * sin(x) - 1"]
    variables_list: List of strings like ["x", "y", "z"]
    """
    # 1. Create symbolic objects
    symbols = sp.symbols(variables_list)

    # 2. Parse the strings into SymPy expressions
    exprs = [sp.parse_expr(eq) for eq in equations_list]

    # 3. Automatically calculate the Jacobian matrix symbolically
    # This is the 'm x n' matrix of partial derivatives
    symbolic_jacobian = sp.Matrix(exprs).jacobian(symbols)

    # 4. Convert SymPy expressions into fast NumPy functions
    # This "compiles" the math so it's as fast as your manual code
    calc_r = sp.lambdify([symbols], exprs, "numpy")
    calc_j = sp.lambdify([symbols], symbolic_jacobian, "numpy")

    return calc_r, calc_j


# --- EXAMPLE USER INPUT ---
user_vars = ["x", "y", "z"]
user_eqs = ["x**2 + y - 5", "z * sin(x) - 1"]

# Initialize the system
calculate_residuals, calculate_jacobian = setup_system(user_eqs, user_vars)

# Now your LM loop stays exactly the same!
beta = np.array([1.0, 1.0, 1.0])
r = np.array(calculate_residuals(beta))
J = np.array(calculate_jacobian(beta))
