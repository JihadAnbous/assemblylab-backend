import jax
import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np


class GeometricSolver:

    def __init__(self, assembly, max_iterations=100, tolerance=1e-6):
        self.assembly = assembly
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.index_map = {}
        self.constraints = []

    def build_index_map(self):
        """
        Build mapping from geometric entities (points, lines) to variable indices.
        Returns the total number of variables.
        """
        self.index_map = {}
        var_index = 0

        # Save all constraints for this assembly to a list
        self.constraints = list(self.assembly.assembly_constraint.all())

        # Collect all unique geometric entities
        entities = set()

        for instance in self.constraints:
            constraint = instance.get_specific_instance()

            # Extract entities based on constraint type
            if hasattr(constraint, "point"):
                entities.add(("point", constraint.point.id))
            if hasattr(constraint, "point1"):
                entities.add(("point", constraint.point1.id))
            if hasattr(constraint, "point2"):
                entities.add(("point", constraint.point2.id))
            if hasattr(constraint, "line"):
                line = constraint.line
                entities.add(("point", line.point1.id))
                entities.add(("point", line.point2.id))
            if hasattr(constraint, "angle"):
                angle = constraint.angle
                entities.add(("point", angle.line1.point1.id))
                entities.add(("point", angle.line1.point2.id))
                entities.add(("point", angle.line2.point1.id))
                entities.add(("point", angle.line2.point2.id))

        # Assign indices to each entity (x and y for points)
        for entity_type, entity_id in sorted(entities):
            if entity_type == "point":
                self.index_map[("point", entity_id, "x")] = var_index
                var_index += 1
                self.index_map[("point", entity_id, "y")] = var_index
                var_index += 1

        return var_index

    def get_initial_variables(self):
        """
        Get initial variable values from current geometry state.
        Returns a JAX array of initial variable values.
        """
        from .models import ReferencePoint

        num_vars = len(self.index_map) // 2  # Each point has x and y
        x0 = np.zeros(len(self.index_map))

        # Populate with current point positions
        for (entity_type, entity_id, coord), idx in self.index_map.items():
            if entity_type == "point":
                point = ReferencePoint.objects.get(id=entity_id)
                if coord == "x":
                    x0[idx] = point.x_plot
                else:  # coord == 'y'
                    x0[idx] = point.y_plot

        return jnp.array(x0)

    def compute_residuals(self, variables):
        """
        Compute residual vector for all constraints.

        Args:
            variables: JAX array of current variable values

        Returns:
            JAX array of residual values
        """
        residuals = []

        for constraint in self.constraints:
            specific = constraint.get_specific_instance()
            if not specific:
                continue

            # Extract variable values based on constraint type
            if specific.type == "FixedPointConstraint":
                x_idx = self.index_map[("point", specific.point.id, "x")]
                y_idx = self.index_map[("point", specific.point.id, "y")]
                r = specific.residual(variables[x_idx], variables[y_idx])
                residuals.append(r)

            elif specific.type == "PointPointCoincidentConstraint":
                x1_idx = self.index_map[("point", specific.point1.id, "x")]
                y1_idx = self.index_map[("point", specific.point1.id, "y")]
                x2_idx = self.index_map[("point", specific.point2.id, "x")]
                y2_idx = self.index_map[("point", specific.point2.id, "y")]
                r = specific.residual(
                    variables[x1_idx],
                    variables[y1_idx],
                    variables[x2_idx],
                    variables[y2_idx],
                )
                residuals.append(r)

            elif specific.type == "PointLineCoincidentConstraint":
                px_idx = self.index_map[("point", specific.point.id, "x")]
                py_idx = self.index_map[("point", specific.point.id, "y")]
                lx1_idx = self.index_map[("point", specific.line.point1.id, "x")]
                ly1_idx = self.index_map[("point", specific.line.point1.id, "y")]
                lx2_idx = self.index_map[("point", specific.line.point2.id, "x")]
                ly2_idx = self.index_map[("point", specific.line.point2.id, "y")]
                r = specific.residual(
                    variables[px_idx],
                    variables[py_idx],
                    variables[lx1_idx],
                    variables[ly1_idx],
                    variables[lx2_idx],
                    variables[ly2_idx],
                )
                residuals.append(r)

            elif specific.type == "DistanceConstraint":
                lx1_idx = self.index_map[("point", specific.line.point1.id, "x")]
                ly1_idx = self.index_map[("point", specific.line.point1.id, "y")]
                lx2_idx = self.index_map[("point", specific.line.point2.id, "x")]
                ly2_idx = self.index_map[("point", specific.line.point2.id, "y")]
                r = specific.residual(
                    variables[lx1_idx],
                    variables[ly1_idx],
                    variables[lx2_idx],
                    variables[ly2_idx],
                )
                residuals.append(r)

            elif specific.type == "AngleConstraint":
                x1_idx = self.index_map[("point", specific.angle.line1.point1.id, "x")]
                y1_idx = self.index_map[("point", specific.angle.line1.point1.id, "y")]
                x2_idx = self.index_map[("point", specific.angle.line1.point2.id, "x")]
                y2_idx = self.index_map[("point", specific.angle.line1.point2.id, "y")]
                x3_idx = self.index_map[("point", specific.angle.line2.point1.id, "x")]
                y3_idx = self.index_map[("point", specific.angle.line2.point1.id, "y")]
                x4_idx = self.index_map[("point", specific.angle.line2.point2.id, "x")]
                y4_idx = self.index_map[("point", specific.angle.line2.point2.id, "y")]
                r = specific.residual(
                    variables[x1_idx],
                    variables[y1_idx],
                    variables[x2_idx],
                    variables[y2_idx],
                    variables[x3_idx],
                    variables[y3_idx],
                    variables[x4_idx],
                    variables[y4_idx],
                )
                residuals.append(r)

        return jnp.array(residuals)

    def compute_jacobian(self, variables):
        """
        Compute Jacobian matrix using automatic differentiation.

        Args:
            variables: JAX array of current variable values

        Returns:
            JAX array representing the Jacobian matrix
        """
        jac_fn = jacfwd(self.compute_residuals)
        return jac_fn(variables)

    def solve(self):
        """
        Solve the constraint system using Newton-Raphson method.

        Returns:
            dict: {
                'success': bool,
                'variables': final variable values,
                'residual_norm': final residual norm,
                'iterations': number of iterations
            }
        """
        # Build index map and get initial values
        num_vars = self.build_index_map()
        variables = self.get_initial_variables()

        if len(self.constraints) == 0:
            return {
                "success": True,
                "variables": variables,
                "residual_norm": 0.0,
                "iterations": 0,
            }

        # Newton-Raphson iteration
        for iteration in range(self.max_iterations):
            # Compute residuals and check convergence
            residuals = self.compute_residuals(variables)
            residual_norm = float(jnp.linalg.norm(residuals))

            if residual_norm < self.tolerance:
                return {
                    "success": True,
                    "variables": variables,
                    "residual_norm": residual_norm,
                    "iterations": iteration,
                }

            # Compute Jacobian
            jacobian = self.compute_jacobian(variables)

            # Solve J * delta = -residuals
            try:
                delta = jnp.linalg.solve(jacobian, -residuals)
            except:
                return {
                    "success": False,
                    "variables": variables,
                    "residual_norm": residual_norm,
                    "iterations": iteration,
                    "error": "Singular Jacobian",
                }

            # Update variables
            variables = variables + delta

        # Max iterations reached
        return {
            "success": False,
            "variables": variables,
            "residual_norm": float(jnp.linalg.norm(self.compute_residuals(variables))),
            "iterations": self.max_iterations,
            "error": "Max iterations reached",
        }

    def update_geometry(self, variables):
        """
        Update the geometry with solved variable values.

        Args:
            variables: JAX array of solved variable values
        """
        from .models import ReferencePoint

        for (entity_type, entity_id, coord), idx in self.index_map.items():
            if entity_type == "point":
                point = ReferencePoint.objects.get(id=entity_id)
                if coord == "x":
                    point.x_plot = float(variables[idx])
                elif coord == "y":
                    point.y_plot = float(variables[idx])
                point.save()
