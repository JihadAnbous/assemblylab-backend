import jax
import jax.numpy as jnp

from django.conf import settings
from django.db import models
from django.core.exceptions import ValidationError

from .mysolver import GeometricSolver

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
    solved = models.BooleanField(default=False, editable=False)

    class Meta:
        verbose_name_plural = "Assemblies"

    def __str__(self):
        return f"{self.name}"

    def solve_geometry(self):
        solver = GeometricSolver(self, max_iterations=100, tolerance=1e-6)
        result = solver.solve()
        if result["success"]:
            # solver.update_geometry(result["variables"])
            print(f"Success: {result['success']}")
