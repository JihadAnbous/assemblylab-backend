import jax
import jax.numpy as jnp

from django.core.exceptions import ValidationError

from django.db import models
from django.apps import apps  # For signals.py

from components.models import Component, Location
from assemblies.models import Assembly

from django_project.utils.helpers import vector

# Create your models here.

REFERENCE_STATUS_CHOICES = [
    ("Fixed", "Fixed"),
    ("Random", "Random"),
    ("Partially constrained", "Partially constrained"),
    ("Fully defined", "Fully defined"),
]


class Reference(models.Model):
    assembly = models.ForeignKey(
        Assembly, on_delete=models.CASCADE, related_name="assembly_reference"
    )
    type = models.CharField(max_length=50, editable=False)
    label = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(
        max_length=21,
        editable=False,
        choices=REFERENCE_STATUS_CHOICES,
        default="Random",
    )
    hidden = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.assembly}-{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def get_specific_instance(self):
        # Get the model class using the type field
        specific_model = apps.get_model(app_label="references", model_name=self.type)
        try:
            instance = specific_model.objects.get(pk=self.pk)
            return instance
        except specific_model.DoesNotExist:
            return None


class ReferenceComponent(Reference):
    component = models.ForeignKey(
        Component, on_delete=models.CASCADE, related_name="component_referencecomponent"
    )

    def transform(self, Sx, Sy, angle):
        # Retrieve all ReferencePoint instances for this ReferenceComponent
        points = ReferencePoint.objects.filter(
            reference_component=self,
        )
        # Initialise the matrix
        coordinates = []
        for point in points:
            coordinates.append([point.x_plot, point.y_plot, 1])
        # Convert to jax array
        component_matrix = jnp.array(coordinates)
        theta = jnp.deg2rad(angle)
        # Transformation matrix
        t = jnp.array(
            [
                [jnp.cos(theta), -jnp.sin(theta), Sx],
                [jnp.sin(theta), jnp.cos(theta), Sy],
                [0, 0, 1],
            ]
        )
        # Transform the component matrix - if error, try matrix, t.T to transpose
        new_matrix = jnp.round(jnp.dot(t, component_matrix), decimals=6)
        # Update the ReferenceComponentPoint fields
        for i, point in enumerate(points):
            point.x_plot = new_matrix[i, 0]
            point.y_plot = new_matrix[i, 1]
        ReferencePoint.objects.bulk_update(points, ["x_plot", "y_plot"])

    def __str__(self):
        return f"{self.assembly}-{self.component}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "ReferenceComponent"
        # Save the instance
        super(ReferenceComponent, self).save(*args, **kwargs)


class ReferencePoint(Reference):
    component = models.ForeignKey(
        ReferenceComponent,
        on_delete=models.CASCADE,
        related_name="component_referencepoint",
        blank=True,
        null=True,
    )
    location = models.ForeignKey(
        Location,
        on_delete=models.CASCADE,
        related_name="location_referencepoint",
        blank=True,
        null=True,
    )
    x_plot = models.FloatField()  # This is never null and are always current
    y_plot = models.FloatField()  # This is never null and are always current

    @property
    def x(self):
        if self.status == "Fixed":
            return self.x_plot
        else:
            return None

    @property
    def y(self):
        if self.status == "Fixed":
            return self.y_plot
        else:
            return None

    # Used for transform function only
    @property
    def matrix(self):
        return jnp.array([self.x_plot, self.y_plot, 1])

    def transform(self, Sx, Sy, angle):
        theta = jnp.deg2rad(angle)
        t = jnp.array(
            [
                [jnp.cos(theta), -jnp.sin(theta), Sx],
                [jnp.sin(theta), jnp.cos(theta), Sy],
                [0, 0, 1],
            ]
        )
        result = jnp.round(jnp.dot(t, self.matrix), decimals=6)
        self.x_plot = result[0]
        self.y_plot = result[1]
        self.save()

    def __str__(self):
        return f"{self.label}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.component and self.location:
            if self.location.component != self.component.component:
                raise ValidationError("This point does not belong to this component.")

    def save(self, *args, **kwargs):
        self.type = "ReferencePoint"
        # Save the instance
        super(ReferencePoint, self).save(*args, **kwargs)


class ReferenceLine(Reference):
    point1 = models.ForeignKey(
        ReferencePoint, on_delete=models.CASCADE, related_name="point1_referenceline"
    )
    point2 = models.ForeignKey(
        ReferencePoint, on_delete=models.CASCADE, related_name="point2_referenceline"
    )

    @property
    def matrix(self):
        point1 = self.point1.matrix
        point2 = self.point1.matrix
        line_vector = vector(point1, point2)
        return line_vector

    def __str__(self):
        return f"{self.assembly}-{self.label}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.point1.assembly != self.assembly:
            raise ValidationError("Point 1 does not belong to this assembly.")
        if self.point2.assembly != self.assembly:
            raise ValidationError("Point 2 does not belong to this assembly.")
        if self.point1 == self.point2:
            raise ValidationError("You cannot make a line out of the same point.")

    def save(self, *args, **kwargs):
        if self.point1.status and self.point2.status == "Fixed":
            self.status = "Fixed"
        self.type = "ReferenceLine"
        # Save the instance
        super(ReferenceLine, self).save(*args, **kwargs)


class ReferenceAngle(Reference):
    line1 = models.ForeignKey(
        ReferenceLine, on_delete=models.CASCADE, related_name="line1_referenceangle"
    )
    line2 = models.ForeignKey(
        ReferenceLine, on_delete=models.CASCADE, related_name="line2_referenceangle"
    )

    def __str__(self):
        return f"{self.assembly}-{self.label}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.line1.assembly != self.assembly:
            raise ValidationError("Line 1 does not belong to this assembly.")
        if self.line2.assembly != self.assembly:
            raise ValidationError("Line 2 does not belong to this assembly.")
        if self.line1 == self.line2:
            raise ValidationError("You cannot make an angle out of the same line.")

    def save(self, *args, **kwargs):
        if self.line1.fixed and self.line2.fixed is True:
            self.fixed = True
        self.type = "ReferenceAngle"
        # Save the instance
        super(ReferenceAngle, self).save(*args, **kwargs)
