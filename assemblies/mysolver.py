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
        index_map = {
                        (point.id, "coordinate"): id,
                        (3, "x"): 0,
                        (3, "y"): 1,
                        (4, "x"): 2,
                        (4, "y"): 3,
                    }
        """
        self.index_map = {}
        self.points = {}
        var_index = 0

        # Get all reference points for this assembly
        references = self.assembly.assembly_reference.filter(type="ReferencePoint")
        points = [instance.get_specific_instance() for instance in references]

        for point in points:
            self.points[point.id] = point
            self.index_map[(point.id, "x")] = var_index
            var_index += 1
            self.index_map[(point.id, "y")] = var_index
            var_index += 1

        return var_index

    def get_initial_variables(self):
        """
        x0 = [5,    6,   -2,    0,   10,   -3]
        Explained below:
             [0]   [1]   [2]   [3]   [4]   [5]  <- index_map index
             pt3_x pt3_y pt4_x pt4_y pt5_x pt5_y
        """
        # Initialise Newton Raphson initial guess array 'x0'.
        x0 = np.zeros(len(self.index_map))
        for (point_id, coordinate), index in self.index_map.items():
            # Retrieve the point instance with the same index as the index_map from the previously created points array
            point = self.points[point_id]
            if coordinate == "x":
                x0[index] = point.x_plot
            else:
                x0[index] = point.y_plot

        return jnp.array(x0)

    def build_residuals(self, variables):
        residuals = []

        # Get all constraints for this assembly
        self.constraints = list(self.assembly.assembly_constraint.all())

        for constraint in self.constraints:
            specific_constraint = constraint.get_specific_instance()

            # Extract variable values from the index map based on the constraint type
            if specific_constraint.type == "FixedPointConstraint":
                x_index = self.index_map[(specific_constraint.point.id, "x")]
                y_index = self.index_map[(specific_constraint.point.id, "y")]
                r = specific_constraint.residual(variables[x_index], variables[y_index])
                residuals.append(r)

            elif specific_constraint.type == "PointPointCoincidentConstraint":
                x1_index = self.index_map[(specific_constraint.point1.id, "x")]
                y1_index = self.index_map[(specific_constraint.point1.id, "y")]
                x2_index = self.index_map[(specific_constraint.point2.id, "x")]
                y2_index = self.index_map[(specific_constraint.point2.id, "y")]
                r = specific_constraint.residual(
                    variables[x1_index],
                    variables[y1_index],
                    variables[x2_index],
                    variables[y2_index],
                )
                residuals.append(r)

            elif specific_constraint.type == "PointLineCoincidentConstraint":
                px_index = self.index_map[(specific_constraint.point.id, "x")]
                py_index = self.index_map[(specific_constraint.point.id, "y")]
                lx1_index = self.index_map[(specific_constraint.line.point1.id, "x")]
                ly1_index = self.index_map[(specific_constraint.line.point1.id, "y")]
                lx2_index = self.index_map[(specific_constraint.line.point2.id, "x")]
                ly2_index = self.index_map[(specific_constraint.line.point2.id, "y")]
                r = specific_constraint.residual(
                    variables[px_index],
                    variables[py_index],
                    variables[lx1_index],
                    variables[ly1_index],
                    variables[lx2_index],
                    variables[ly2_index],
                )
                residuals.append(r)

            elif specific_constraint.type == "DistanceConstraint":
                x1_index = self.index_map[(specific_constraint.line.point1.id, "x")]
                y1_index = self.index_map[(specific_constraint.line.point1.id, "y")]
                x2_index = self.index_map[(specific_constraint.line.point2.id, "x")]
                y2_index = self.index_map[(specific_constraint.line.point2.id, "y")]
                r = specific_constraint.residual(
                    variables[x1_index],
                    variables[y1_index],
                    variables[x2_index],
                    variables[y2_index],
                )
                residuals.append(r)

            elif specific_constraint.type == "AngleConstraint":
                x1_index = self.index_map[
                    (specific_constraint.angle.line1.point1.id, "x")
                ]
                y1_index = self.index_map[
                    (specific_constraint.angle.line1.point1.id, "y")
                ]
                x2_index = self.index_map[
                    (specific_constraint.angle.line1.point2.id, "x")
                ]
                y2_index = self.index_map[
                    (specific_constraint.angle.line1.point2.id, "y")
                ]
                x3_index = self.index_map[
                    (specific_constraint.angle.line2.point1.id, "x")
                ]
                y3_index = self.index_map[
                    (specific_constraint.angle.line2.point1.id, "y")
                ]
                x4_index = self.index_map[
                    (specific_constraint.angle.line2.point2.id, "x")
                ]
                y4_index = self.index_map[
                    (specific_constraint.angle.line2.point2.id, "y")
                ]
                r = specific_constraint.residual(
                    variables[x1_index],
                    variables[y1_index],
                    variables[x2_index],
                    variables[y2_index],
                    variables[x3_index],
                    variables[y3_index],
                    variables[x4_index],
                    variables[y4_index],
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
