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
    type = models.CharField(max_length=50, editable=False)

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


class ParallelConstraint(Constraint):
    line1 = models.ForeignKey(
        ReferenceLine, on_delete=models.CASCADE, related_name="line1_parallelconstraint"
    )
    line2 = models.ForeignKey(
        ReferenceLine, on_delete=models.CASCADE, related_name="line2_parallelconstraint"
    )

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "ParallelConstraint"
        # Save the instance
        super(ParallelConstraint, self).save(*args, **kwargs)


class PerpendicularConstraint(Constraint):
    line1 = models.ForeignKey(
        ReferenceLine,
        on_delete=models.CASCADE,
        related_name="line1_perpendicularconstraint",
    )
    line2 = models.ForeignKey(
        ReferenceLine,
        on_delete=models.CASCADE,
        related_name="line2_perpendicularconstraint",
    )

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "PerpendicularConstraint"
        # Save the instance
        super(PerpendicularConstraint, self).save(*args, **kwargs)


class VariableDistanceConstraint(Constraint):
    line = models.ForeignKey(
        ReferenceLine,
        on_delete=models.CASCADE,
        related_name="line_variabledistanceconstraint",
    )
    minimum = models.FloatField(validators=[MinValueValidator(0.0)])
    maximum = models.FloatField(validators=[MinValueValidator(0.0)])

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "VariableDistanceConstraint"
        # Save the instance
        super(VariableDistanceConstraint, self).save(*args, **kwargs)


class VariableAngleConstraint(Constraint):
    angle = models.ForeignKey(
        ReferenceAngle,
        on_delete=models.CASCADE,
        related_name="angle_variableangleconstraint",
    )
    minimum = models.FloatField(validators=[MinValueValidator(0.0)])
    maximum = models.FloatField(validators=[MinValueValidator(0.0)])

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "VariableAngleConstraint"
        # Save the instance
        super(VariableAngleConstraint, self).save(*args, **kwargs)
