import jax
import jax.numpy as jnp

from django.conf import settings
from django.db import models
from django.core.exceptions import ValidationError

from components.models import Component, Location, UNITS_CHOICES

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
    transformation_point = models.ForeignKey(
        Location,
        on_delete=models.CASCADE,
        related_name="transformationpoint_assemblycomponent",
    )
    x_translation = models.FloatField(blank=True, null=True)
    y_translation = models.FloatField(blank=True, null=True)
    rotation = models.FloatField(blank=True, null=True)

    @property
    def matrix(self):
        matrix = jnp.array([0])
        return matrix

    def __str__(self):
        return f"{self.assembly}-{self.component}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.transformation_point.component != self.component:
            raise ValidationError(
                "This transformation point does not belong to this component."
            )

    def save(self, *args, **kwargs):
        if self.x_translation is None:
            self.x_translation = 0
        if self.y_translation is None:
            self.y_translation = 0
        if self.rotation is None:
            self.rotation = 0
        # If 2 points are fixed then self.fixed = True
        # Save the instance
        super(AssemblyComponent, self).save(*args, **kwargs)
