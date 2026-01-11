import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass
from jax import grad, jacfwd
from sympy import Symbol, diff

# Levenberg-Marquardt method (LM method)
# `python -m assemblies.utils.solver`


# Points
@dataclass
class Point:
    x: float
    y: float
    label: str
    id: int


# Test points
A = Point(x=-214, y=44, label="A", id=1)
B = Point(x=3, y=90, label="B", id=7)
C = Point(x=98, y=-245, label="C", id=11)

points = (A, B, C)
print(f"Points:")
for p in points:
    print(p.label, ": (", p.x, ",", p.y, ")")
point_map = {}
for i, p in enumerate(points):
    point_map[p.id] = i
print(f"Point map:\n", point_map)

"""Above has been moved to point_map property field of Assembly model"""

# Variables array - FOR DISPLAYING VARIABLES ONLY
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


"""Above has been moved as property fields to Constraint models"""

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

"""Above has been moved as property fields to Assembly model"""

# Create the Jacobian matrix
rows = len(constraints)
cols = len(points) * 2
jacobian = np.zeros((rows, cols))

# "sparse_row" is each dictionary instance in jacobian_map: (e.g., {0: 1, 2: -1})
# "value" is the derivative value of each dictionary instance
for row_index, sparse_row in enumerate(jacobian_map):
    for col_index, value in sparse_row.items():
        jacobian[row_index, col_index] = value

print("Jacobian (NumPy):\n", jacobian)

"""Above has been moved as property fields to Assembly model"""

"""
Solver section
"""

# Initial setup
lam = 0.1  # Damping factor
tol = 1e-8  # Convergence tolerance
max_iter = 50  # Maximum iterations

"""Initial guess"""
# Initialise
beta_list = []
for p in points:
    beta_list.append(p.x)
    beta_list.append(p.y)
# Convert to NumPy array
beta = np.array(beta_list)
# Ensure they are floats - delete when connected to Django as they are floatfields
# beta = beta.astype(float)
print(f"Initial beta array: {beta}")

"""Solver loop"""
for i in range(max_iter):
    # Dictionary that maps the variables with their values
    variable_map = {}
    for index, symbol in enumerate(vars):
        # {x1: 24, y1: -45, etc.}
        variable_map[symbol] = beta[index]

    # Calculate the residuals
    residuals = []
    for expression in constraints:
        # expression.subs(old, new)
        # Swaps each constraint's variable symbol with their respective values in variable_map
        numerical_expression = expression.subs(variable_map)
        # Convert the SymPy result to a standard float
        numerical_value = float(numerical_expression)
        residuals.append(numerical_value)
    residuals = np.array(residuals)

    # Calculate J transposed times J
    # This creates a square matrix (Hessian approximation)
    jtj = jacobian.T @ jacobian

    # Add the damping factor (lambda) to the diagonal
    damping_matrix = lam * np.eye(cols)
    lhs = jtj + damping_matrix

    # Calculate J transposed times the residuals
    rhs = -jacobian.T @ residuals

    # (J^T J + λI) * delta_beta = -J^T * r -> Solve for delta_beta
    delta_beta = np.linalg.solve(lhs, rhs)

    # Update current guess with delta_beta
    beta = beta + delta_beta

    # Determine magnitude for convergence
    magnitude = np.linalg.norm(delta_beta)
    print(f"Iteration {i}: Magnitude = {magnitude}")

    # If magnitude < tolerance, converged
    if magnitude < tol:
        print("Solver has converged.")
        break

"""Update points"""
for i, p in enumerate(points):
    p.x = beta[2 * i]
    p.y = beta[2 * i + 1]

print(f"Updated points:")
for p in points:
    print(p.label, ": (", p.x, ",", p.y, ")")
