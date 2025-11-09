from django.db import models
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator

from assemblies.models import Assembly, AssemblyPoint
from references.models import ReferenceLine, ReferenceAngle

# Create your models here.


class Constraint(models.Model):
    assembly = models.ForeignKey(
        Assembly, on_delete=models.CASCADE, related_name="assembly_constraint"
    )
    type = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.assembly}-{self.type}"

    def clean(self):
        super().clean()
        # Additional validation


class CoincidentConstraint(Constraint):
    point1 = models.ForeignKey(
        AssemblyPoint,
        on_delete=models.CASCADE,
        related_name="point1_coincidentconstraint",
    )
    point2 = models.ForeignKey(
        AssemblyPoint,
        on_delete=models.CASCADE,
        related_name="point2_coincidentconstraint",
    )

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.point1 == self.point2:
            raise ValidationError(
                "The selected point is already coincident with itself."
            )

    def save(self, *args, **kwargs):
        self.type = "CoincidentConstraint"
        # Save the instance
        super(CoincidentConstraint, self).save(*args, **kwargs)


class DistanceConstraint(Constraint):
    line = models.ForeignKey(
        ReferenceLine, on_delete=models.CASCADE, related_name="line_distanceconstraint"
    )
    value = models.FloatField(validators=[MinValueValidator(0.0)])

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "DistanceConstraint"
        # Save the instance
        super(DistanceConstraint, self).save(*args, **kwargs)


class AngleConstraint(Constraint):
    angle = models.ForeignKey(
        ReferenceAngle, on_delete=models.CASCADE, related_name="angle_angleconstraint"
    )
    value = models.FloatField(help_text="Degrees")

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "AngleConstraint"
        # Save the instance
        super(AngleConstraint, self).save(*args, **kwargs)


def constraint_parallel(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y]
    # vector2 must be a jax 1D array: [x, y]
    # Constraint satisfied when function = 0
    function = cross_product(vector1, vector2)
    return function


def constraint_perpendicular(vector1, vector2):
    # vector1 must be a jax 1D array: [x, y]
    # vector2 must be a jax 1D array: [x, y]
    # Constraint satisfied when function = 0
    function = dot_product(vector1, vector2)
    return function
