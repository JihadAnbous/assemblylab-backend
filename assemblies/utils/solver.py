import numpy as np
from sympy import Symbol


def run_solver(assembly):
    from references.models import ReferencePoint

    # 1. Gather structural data
    constrained_points = assembly.constrained_points
    point_map = assembly.point_map
    constraints_expressions = assembly.constraints

    # Pre-fetch the symbolic Jacobian map once (the symbolic derivatives don't change)
    # This assumes assembly.jacobian_map returns: [{col_idx: expression}, ...]
    symbolic_jacobian_map = assembly.jacobian_map

    # 2. Setup variables
    # We create a list of SymPy symbols that correspond to our beta indices
    vars_symbols = []
    for p in constrained_points:
        vars_symbols.append(Symbol(f"x{p.id}"))
        vars_symbols.append(Symbol(f"y{p.id}"))

    # 3. Initial guess (beta)
    beta = np.array([p.x_plot for p in constrained_points for p in [p]], dtype=float)
    # Flattening x,y into beta: [x1, y1, x2, y2...]
    beta_list = []
    for p in constrained_points:
        beta_list.extend([float(p.x_plot), float(p.y_plot)])
    beta = np.array(beta_list)

    # Constants
    lam = 0.1
    tol = 1e-10
    max_iter = 50
    num_vars = len(beta)
    num_constraints = len(constraints_expressions)

    for i in range(max_iter):
        # Create the mapping for SymPy substitution using CURRENT beta values
        variable_map = {vars_symbols[k]: beta[k] for k in range(num_vars)}

        # --- CALCULATE RESIDUALS (r) ---
        residuals = np.zeros(num_constraints)
        for idx, expr in enumerate(constraints_expressions):
            residuals[idx] = float(expr.subs(variable_map))

        # --- CALCULATE JACOBIAN (J) ---
        # We build the matrix based on current beta values
        J = np.zeros((num_constraints, num_vars))
        for row_idx, sparse_row in enumerate(symbolic_jacobian_map):
            for col_idx, symbolic_deriv in sparse_row.items():
                # Substitute current beta values into the derivative expression
                J[row_idx, col_idx] = float(symbolic_deriv.subs(variable_map))

        # --- LEVENBERG-MARQUARDT STEP ---
        jtj = J.T @ J
        damping = lam * np.eye(num_vars)
        rhs = -J.T @ residuals

        try:
            delta_beta = np.linalg.solve(jtj + damping, rhs)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered.")
            break

        beta += delta_beta

        magnitude = np.linalg.norm(delta_beta)
        if magnitude < tol:
            print(f"Converged in {i} iterations.")
            break

    # 4. Final Database Update
    points_to_update = []
    for idx, p in enumerate(constrained_points):
        p_db = ReferencePoint.objects.get(id=p.id)
        p_db.x_plot = beta[2 * idx]
        p_db.y_plot = beta[2 * idx + 1]
        points_to_update.append(p_db)

    ReferencePoint.objects.bulk_update(points_to_update, ["x_plot", "y_plot"])
    return beta
