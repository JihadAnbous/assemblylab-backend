import numpy as np
from sympy import Symbol, diff

from dataclasses import dataclass

"""Levenberg-Marquardt method solver"""


def run_solver(assembly):
    constrained_points = assembly.constrained_points
    constraints = assembly.constraints
    point_map = assembly.point_map
    jacobian = assembly.jacobian
    print("Created jacobian matrix.")

    vars = [0] * len(constrained_points) * 2
    for i, p in enumerate(constrained_points):
        vars[2 * i] = Symbol(f"x{p.id}")
        vars[2 * i + 1] = Symbol(f"y{p.id}")
    # Initial setup
    lam = 0.1  # Damping factor
    tol = 1e-8  # Convergence tolerance
    max_iter = 50  # Maximum iterations
    print("Variables", vars)

    """Initial guess"""
    # Initialise
    beta_list = []
    for p in constrained_points:
        beta_list.append(p.x_plot)
        beta_list.append(p.y_plot)
    # Convert to NumPy array
    beta = np.array(beta_list)
    print(f"Initial beta array: {beta_list}")

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
        cols = len(point_map) * 2
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
            print("Solver has converged.", beta)
            break

    """Update constrained_points"""
    from references.models import ReferencePoint

    points_to_update = []
    for i, p in enumerate(constrained_points):
        x = beta[2 * i]
        y = beta[2 * i + 1]
        point = ReferencePoint.objects.get(id=p.id)
        point.x_plot = x
        point.y_plot = y
        points_to_update.append(point)

    ReferencePoint.objects.bulk_update(points_to_update, ["x_plot", "y_plot"])

    """Transform all components"""

    return beta
