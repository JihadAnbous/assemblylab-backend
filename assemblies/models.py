import jax
import jax.numpy as jnp

from django.conf import settings
from django.db import models
from django.core.exceptions import ValidationError

from components.models import Component, Location, UNITS_CHOICES
from references.models import ReferenceComponentPoint

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


class AssemblyComponent(models.Model):
    assembly = models.ForeignKey(
        Assembly, on_delete=models.CASCADE, related_name="assembly_assemblycomponent"
    )
    component = models.ForeignKey(
        Component, on_delete=models.CASCADE, related_name="component_assemblycomponent"
    )
    fixed = models.BooleanField(default=False)

    def transform(self, Sx, Sy, angle, rotate_about):
        # Retrieve all ReferenceComponentPoint instances for this AssemblyComponent
        points = ReferenceComponentPoint.objects.filter(
            assembly=self.assembly,
            component=self.component,
        )
        # Initialise the matrix
        coordinates = []
        for point in points:
            coordinates.append([point.x_plot, point.y_plot, 1])
        # Convert to jax array
        component_matrix = jnp.array(coordinates)
        # Transformation matrix
        t = jnp.array(
            [
                [jnp.cos(angle), -jnp.sin(angle), Sx],
                [jnp.sin(angle), jnp.cos(angle), Sy],
                [0, 0, 1],
            ]
        )
        # Transform the component matrix - if error, try matrix, t.T to transpose
        new_matrix = jnp.round(jnp.dot(t, component_matrix), decimals=6)
        # Update the ReferenceComponentPoint fields
        for i, point in enumerate(points):
            point.x_plot = new_matrix[i, 0]
            point.y_plot = new_matrix[i, 1]
        ReferenceComponentPoint.objects.bulk_update(points, ["x_plot", "y_plot"])

    def __str__(self):
        return f"{self.assembly}-{self.component}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        # If 2 points are fixed then self.fixed = True
        # Save the instance
        super(AssemblyComponent, self).save(*args, **kwargs)
