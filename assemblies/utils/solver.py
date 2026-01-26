import numpy as np
from sympy import Symbol


def run_solver(assembly):
    from references.models import ReferencePoint

    # Gather data
    constrained_points = assembly.constrained_points
    constraints_expressions = assembly.constraints

    # Fetch the Jacobian map
    jacobian_map = assembly.jacobian_map

    # Setup variables
    # We create a list of SymPy symbols that correspond to our beta indices
    vars_symbols = []
    for p in constrained_points:
        vars_symbols.append(Symbol(f"x{p.id}"))
        vars_symbols.append(Symbol(f"y{p.id}"))

    # Initial guess
    # Flattening x,y into beta: [x1, y1, x2, y2...]
    beta_list = []
    for p in constrained_points:
        beta_list.extend([float(p.x_plot), float(p.y_plot)])
    beta = np.array(beta_list)

    # --- NEW: INITIAL SATISFACTION CHECK ---
    # Check if the points already satisfy constraints before doing any work
    initial_map = {vars_symbols[k]: beta[k] for k in range(len(beta))}
    initial_residuals = [
        float(expr.subs(initial_map)) for expr in constraints_expressions
    ]

    if initial_residuals and np.max(np.abs(initial_residuals)) < 1e-6:
        # Everything is already correct; do nothing and exit
        return beta

    # Constants
    lam = 0.1
    tol = 1e-10
    max_iter = 50
    num_vars = len(beta)
    num_constraints = len(constraints_expressions)

    for i in range(max_iter):
        magnitude = None
        # Create the mapping for SymPy substitution using CURRENT beta values
        variable_map = {vars_symbols[k]: beta[k] for k in range(num_vars)}

        # --- CALCULATE RESIDUALS (r) ---
        residuals = np.zeros(num_constraints)
        for idx, expr in enumerate(constraints_expressions):
            residuals[idx] = float(expr.subs(variable_map))

        # Compute the sum of residual errors
        current_error = np.sum(residuals**2)

        # --- CALCULATE JACOBIAN (J) ---
        # We build the matrix based on current beta values
        J = np.zeros((num_constraints, num_vars))
        for row_idx, sparse_row in enumerate(jacobian_map):
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
            print("Singular matrix encountered. Increasing damping.")
            lam *= 10
            continue

        # --- LAMBDA UPDATE STEP ---
        # Test the step before committing to it
        test_beta = beta + delta_beta
        test_variable_map = {vars_symbols[k]: test_beta[k] for k in range(num_vars)}

        test_residuals = np.zeros(num_constraints)
        for idx, expr in enumerate(constraints_expressions):
            test_residuals[idx] = float(expr.subs(test_variable_map))

        new_error = np.sum(test_residuals**2)

        if new_error < current_error:
            # Step is successful: decrease damping and move beta
            lam /= 10
            beta = test_beta
            magnitude = np.linalg.norm(delta_beta)
            if magnitude < tol:
                print(f"Converged in {i} iterations.")
                break
        else:
            # Step is unsuccessful: increase damping and try again from current beta
            lam *= 10
            if lam > 1e12:  # Safety break to prevent infinite loops on impossible math
                print("Damping grew too large. Stopping.")
                break

    # --- FINAL CONTRADICTION CHECK ---
    # We check if the residuals are actually small after the loop
    final_variable_map = {vars_symbols[k]: beta[k] for k in range(num_vars)}
    final_residuals = [
        float(expr.subs(final_variable_map)) for expr in constraints_expressions
    ]
    max_res = np.max(np.abs(final_residuals))

    if max_res > 1e-4:
        print(
            f"WARNING: Solver stopped with high residuals (Max: {max_res}). Possible contradiction."
        )

    # Print results for debugging
    print(f"beta:", beta)
    print(f"variable_map:", variable_map)
    print(f"final_variable_map:", final_variable_map)
    print(f"residuals:", residuals)
    print(f"final_residuals:", final_residuals)
    print(f"current_error:", current_error)
    print(f"jtj:", jtj)
    print(f"damping:", damping)
    print(f"rhs:", rhs)
    print(f"delta_beta:", delta_beta)
    if magnitude is not None:
        print(f"Iteration {i} magnitude: {magnitude}")
    print(f"lam:", lam)
    print(f"max_res:", max_res)

    # Gather the points that need updating
    points_to_update = []
    for idx, p in enumerate(constrained_points):
        p_db = ReferencePoint.objects.get(id=p.id)
        p_db.x_plot = beta[2 * idx]
        p_db.y_plot = beta[2 * idx + 1]
        points_to_update.append(p_db)
    # Bulk update the database
    ReferencePoint.objects.bulk_update(points_to_update, ["x_plot", "y_plot"])
    return beta


"""
I think the solver is now finished.
You must next figure out how to flag when the assembly is over-constrained. 
Maybe DOF can help. DOF of each component/point or of entire assembly???

Don't let assembly save if it is over-constrained.

Then start transforming the components. 

Transforming: 
If 2 component points have been solved, get SX, Sy and theta. Apply transform to all OTHER points of the cmponent.

If 1 component point has been solved only, get Sx, Sy only. Apply transform to all OTHER points of the component.

For studies:
Make a new solver that instead returns x's and y's - instead of updating x_plot and y_plot.
If user wishes to see them, simple equate the x_plot and y_plots to the found x's and y's.
"""
