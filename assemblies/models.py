import jax
import jax.numpy as jnp

from django.conf import settings
from django.db import models
from django.core.exceptions import ValidationError

from components.models import Component, Location, UNITS_CHOICES

# Create your models here.


class Assembly(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
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
        # Save the instance
        super(AssemblyComponent, self).save(*args, **kwargs)


class AssemblyPoint(models.Model):
    component = models.ForeignKey(
        AssemblyComponent,
        on_delete=models.CASCADE,
        related_name="component_assemblypoint",
    )
    location = models.ForeignKey(
        Location, on_delete=models.CASCADE, related_name="location_assemblypoint"
    )
    label = models.CharField(max_length=32)
    fixed = models.BooleanField(default=False, editable=False)
    x = models.FloatField(blank=True, null=True)
    y = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{self.component}-{self.label}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.location.component != self.component.component:
            raise ValidationError("This point does not belong to this component.")

    def save(self, *args, **kwargs):
        if self.component.fixed is True:
            self.fixed = True
        if self.fixed is True:
            # Try to fix self.x and self.y to the assembly component's matrix
            pass
        # Save the instance
        super(AssemblyPoint, self).save(*args, **kwargs)
