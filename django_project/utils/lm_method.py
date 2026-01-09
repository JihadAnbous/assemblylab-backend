import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass
from jax import grad, jacfwd
from sympy import Symbol, diff

# Levenberg-Marquardt method (LM method)
# `python -m django_project.utils.lm_method`


# Points
@dataclass
class Point:
    x: float
    y: float
    label: str
    id: int


# Test points
A = Point(x=0, y=0, label="A", id=1)
B = Point(x=0, y=0, label="B", id=7)
C = Point(x=0, y=0, label="C", id=11)

points = (A, B, C)
point_map = {}
for i, p in enumerate(points):
    point_map[p.id] = i
print(f"Point map:\n", point_map)

# Variables array - MOST LIKELY DO NOT NEED BUT DOING IT ANYWAY
vars = [0] * len(points) * 2
for i, p in enumerate(points):
    vars[2 * i] = Symbol(f"x{p.id}")
    vars[2 * i + 1] = Symbol(f"y{p.id}")
print(f"Variables:\n", vars)


# Constraints
@dataclass
class CoincidentConstraint:
    point_1: Point
    point_2: Point
    point_map: dict

    @property
    def symbols(self):
        """Returns the specific SymPy symbols for this constraint."""
        return (
            Symbol(f"x{self.point_1.id}"),
            Symbol(f"x{self.point_2.id}"),
            Symbol(f"y{self.point_1.id}"),
            Symbol(f"y{self.point_2.id}"),
        )

    @property
    def r(self):
        """The symbolic residual expressions for both x and y."""
        x1, x2, y1, y2 = self.symbols
        return [x1 - x2, y1 - y2]

    @property
    def j(self):
        """Returns a list of dictionaries, one for each residual row."""
        x1, x2, y1, y2 = self.symbols
        r_x, r_y = self.r

        # Mapping indices for x (2*i) and y (2*i + 1)
        idx1_x = 2 * self.point_map[self.point_1.id]
        idx2_x = 2 * self.point_map[self.point_2.id]
        idx1_y = 2 * self.point_map[self.point_1.id] + 1
        idx2_y = 2 * self.point_map[self.point_2.id] + 1

        return [
            {idx1_x: diff(r_x, x1), idx2_x: diff(r_x, x2)},  # Row for x residual
            {idx1_y: diff(r_y, y1), idx2_y: diff(r_y, y2)},  # Row for y residual
        ]


# Create constraints and jacobian_map
c1 = CoincidentConstraint(A, B, point_map)
c2 = CoincidentConstraint(B, C, point_map)
constraints = []
jacobian_map = []
for c in [c1, c2]:
    constraints.extend(c.r)
    jacobian_map.extend(c.j)

print(f"Constraints/residuals:\n", constraints)
print(f"Jacobian map:\n", jacobian_map)

# Create the Jacobian matrix
rows = len(constraints)
cols = len(points) * 2
jacobian = np.zeros((rows, cols))

# Sparse row is each dictionary instance in jacobian_map: (e.g., {0: 1, 2: -1})
for row_index, sparse_row in enumerate(jacobian_map):
    for col_index, value in sparse_row.items():
        jacobian[row_index, col_index] = value

print("Jacobian (NumPy):\n", jacobian)

# # Jacobian
# # rows = total number of variables = n
# rows = len(point_map)
# # columns = total number of constraints = m
# columns = len(constraints)

# # 1. Initialize a matrix of zeros
# jacobian = np.zeros((columns, rows))

# # 2. Populate the matrix
# for row_index, sparse_row in enumerate(jacobian_map):
#     for col_index, value in sparse_row.items():
#         jacobian[row_index, col_index] = value

# print("Jacobian (NumPy):\n", jacobian)

""" 
Errors so far:
- constraints do not include y1 - y2
- jacobian does not include y values... do we need them?
"""

# # Residuals
# def calculate_residuals(beta):
#     x, y, z = beta
#     r1 = x**2 + y - 5
#     r2 = z * np.sin(x) - 1
#     return np.array([r1, r2])


# # Jacobian [m x n] matrix
# # m : number of residuals | rows
# # n : number of variables | columns
# def calculate_jacobian(beta):
#     x, y, z = beta
#     # Row 1 derivatives
#     dr1_dx = 2 * x
#     dr1_dy = 1
#     dr1_dz = 0

#     # Row 2 derivatives
#     dr2_dx = z * np.cos(x)
#     dr2_dy = 0
#     dr2_dz = np.sin(x)

#     return np.array([[dr1_dx, dr1_dy, dr1_dz], [dr2_dx, dr2_dy, dr2_dz]])


# # Initial setup
# beta = np.array([1.0, 1.0, 1.0])  # Initial guess [x, y, z]
# lam = 0.1  # Damping factor
# tol = 1e-8  # Convergence tolerance
# max_iter = 50  # Maximum iterations

# # Solver
# # Loop through the maximum iterations
# for i in range(max_iter):
#     r = calculate_residuals(beta)
#     J = calculate_jacobian(beta)

#     # Sum of Squares - Sum of residual errors
#     S_old = np.sum(r**2)

#     # (J^T J + lambda * I) * delta = J^T * r
#     jtj = J.T @ J
#     gradient = J.T @ r

#     # The LM update (Damping)
#     H_damped = jtj + lam * np.eye(len(beta))

#     try:
#         # H_damped * delta_beta = gradient > np.linalg.solve() finds delta_beta
#         delta_beta = np.linalg.solve(H_damped, gradient)
#     except np.linalg.LinAlgError:
#         # If the matrix is still singular/non-invertible, increase damping
#         lam = lam * 10
#         continue  # Skip the rest of this iteration

#     # Test the new position
#     beta_new = beta - delta_beta
#     r_new = calculate_residuals(beta_new)
#     S_new = np.sum(r_new**2)

#     if S_new < S_old:
#         # Success!
#         print(f"Iter {i}: Error {S_new:.6f}, lam {lam:.4f}")
#         beta = beta_new
#         lam = lam / 10  # Get faster
#         if np.linalg.norm(delta_beta) < tol:
#             print("Converged!")
#             break
#     else:
#         # Failure!
#         lam = lam * 10  # Get more stable

# print(f"\nFinal Parameters:\nx: {beta[0]:.4f}\ny: {beta[1]:.4f}\nz: {beta[2]:.4f}")


# # Automating it with sympy


# def setup_system(equations_list, variables_list):
#     """
#     equations_list: List of strings like ["x**2 + y - 5", "z * sin(x) - 1"]
#     variables_list: List of strings like ["x", "y", "z"]
#     """
#     # 1. Create symbolic objects
#     symbols = sp.symbols(variables_list)

#     # 2. Parse the strings into SymPy expressions
#     exprs = [sp.parse_expr(eq) for eq in equations_list]

#     # 3. Automatically calculate the Jacobian matrix symbolically
#     # This is the 'm x n' matrix of partial derivatives
#     symbolic_jacobian = sp.Matrix(exprs).jacobian(symbols)

#     # 4. Convert SymPy expressions into fast NumPy functions
#     # This "compiles" the math so it's as fast as your manual code
#     calc_r = sp.lambdify([symbols], exprs, "numpy")
#     calc_j = sp.lambdify([symbols], symbolic_jacobian, "numpy")

#     return calc_r, calc_j


# # --- EXAMPLE USER INPUT ---
# user_vars = ["x", "y", "z"]
# user_eqs = ["x**2 + y - 5", "z * sin(x) - 1"]

# # Initialize the system
# calculate_residuals, calculate_jacobian = setup_system(user_eqs, user_vars)

# # Now your LM loop stays exactly the same!
# beta = np.array([1.0, 1.0, 1.0])
# r = np.array(calculate_residuals(beta))
# J = np.array(calculate_jacobian(beta))
