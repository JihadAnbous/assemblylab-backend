from dataclasses import dataclass
from jax import grad, jacfwd
from sympy import Symbol, diff
import numpy as np

from django.conf import settings
from django.db import models
from django.core.exceptions import ValidationError

# from .utils.solver import run_solver
from .utils.solver import run_solver

# Create your models here.

ASSEMBLY_STATUS_CHOICES = [
    ("Not constrained", "Not constrained"),
    ("Partially constrained", "Partially constrained"),
    ("Fully constrained", "Fully constrained"),
]


class Assembly(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=100, choices=ASSEMBLY_STATUS_CHOICES, editable=False
    )

    class Meta:
        verbose_name_plural = "Assemblies"

    def __str__(self):
        return f"{self.name}"

    @property
    def constrained_points(self):
        from constraints.models import (
            FixedPointConstraint,
            PointPointCoincidentConstraint,
            DistanceConstraint,
        )

        unique_nodes = []
        seen_ids = set()

        # Define the logic for extracting points from different constraint types
        for constraint in FixedPointConstraint.objects.filter(
            assembly=self
        ).select_related("point"):
            self._add_unique_point(constraint.point, unique_nodes, seen_ids)

        for constraint in PointPointCoincidentConstraint.objects.filter(
            assembly=self
        ).select_related("point1", "point2"):
            self._add_unique_point(constraint.point1, unique_nodes, seen_ids)
            self._add_unique_point(constraint.point2, unique_nodes, seen_ids)

        # Handle DistanceConstraint via the nested 'line' object
        # select_related('line__point1', 'line__point2') optimizes the DB query
        for constraint in DistanceConstraint.objects.filter(
            assembly=self
        ).select_related("line__point1", "line__point2"):
            self._add_unique_point(constraint.line.point1, unique_nodes, seen_ids)
            self._add_unique_point(constraint.line.point2, unique_nodes, seen_ids)

        return unique_nodes

    def _add_unique_point(self, point, unique_nodes, seen_ids):
        """Helper to avoid duplicating code"""
        if point and point.id not in seen_ids:
            unique_nodes.append(point)
            seen_ids.add(point.id)

    @property
    def point_map(self):
        result = {}
        for i, p in enumerate(self.constrained_points):
            result[p.id] = i
        return result

    @property
    def constraints(self):
        """
        Returns the mathematical residuals (r) for all specific constraints.
        """
        from constraints.models import Constraint

        # Get all base constraints
        base_constraints = Constraint.objects.filter(assembly=self)

        result = []
        for base in base_constraints:
            specific = base.get_specific_instance()
            if specific and hasattr(specific, "r"):
                result.extend(specific.r)
        return result

    @property
    def jacobian_map(self):
        """
        Returns the jacobian (j) for all specific constraints.
        """
        from constraints.models import Constraint

        # Get all base constraints
        base_constraints = Constraint.objects.filter(assembly=self)

        result = []
        for base in base_constraints:
            specific = base.get_specific_instance()
            if specific and hasattr(specific, "j"):
                result.extend(specific.j)
        return result

    @property
    def solve(self):
        results = run_solver(self)
        return results

    # @property
    # def jacobian(self):
    #     rows = len(self.constraints)
    #     cols = len(self.point_map) * 2
    #     result = np.zeros((rows, cols))

    #     # "sparse_row" is each dictionary instance in jacobian_map: (e.g., {0: 1, 2: -1})
    #     # "value" is the derivative value of each dictionary instance
    #     for row_index, sparse_row in enumerate(self.jacobian_map):
    #         for col_index, value in sparse_row.items():
    #             result[row_index, col_index] = value

    #     return result
