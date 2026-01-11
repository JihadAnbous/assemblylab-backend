import jax
import jax.numpy as jnp

from django.conf import settings
from django.db import models
from django.core.exceptions import ValidationError

from .utils import solver

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

    @property
    def point_map(self):
        # Import here to avoid circular imports
        from constraints.models import (
            FixedPointConstraint,
            PointPointCoincidentConstraint,
        )

        # A node is a point/line element
        unique_nodes = []
        seen_ids = (
            set()
        )  # Data type to store an unordered collection of unique, immutable elements

        # [(model_class, point_fields)]
        constraint_configs = [
            (FixedPointConstraint, ["point"]),
            (PointPointCoincidentConstraint, ["point1", "point2"]),
        ]

        for model_class, point_fields in constraint_configs:
            nodes = model_class.objects.filter(assembly=self).select_related(
                *point_fields
            )

            for node in nodes:
                for field_name in point_fields:
                    p = getattr(node, field_name)
                    if p and p.id not in seen_ids:
                        unique_nodes.append(p)
                        seen_ids.add(p.id)
        result = {}
        for i, p in enumerate(unique_nodes):
            result[p.id] = i
        return result
