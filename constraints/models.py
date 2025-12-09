import jax
import jax.numpy as jnp

from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator

from django.db import models
from django.apps import apps

from assemblies.models import Assembly
from references.models import Reference, ReferencePoint, ReferenceLine, ReferenceAngle

from django_project.utils.helpers import dot_product, vector_magnitude

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

    def get_specific_instance(self):
        # Get the model class using the type field
        specific_model = apps.get_model(app_label="constraints", model_name=self.type)
        try:
            instance = specific_model.objects.get(pk=self.pk)
            return instance
        except specific_model.DoesNotExist:
            return None


class FixedPointConstraint(Constraint):
    point = models.OneToOneField(
        ReferencePoint,
        on_delete=models.CASCADE,
        related_name="point_fixedpointconstraint",
    )

    def residual(self, x, y):
        x_plot = self.point.x_plot
        y_plot = self.point.y_plot
        result = (x - x_plot) ** 2 + (y - y_plot) ** 2
        return result

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "FixedPointConstraint"
        # Save the instance
        super(FixedPointConstraint, self).save(*args, **kwargs)


class PointPointCoincidentConstraint(Constraint):
    point1 = models.ForeignKey(
        ReferencePoint,
        on_delete=models.CASCADE,
        related_name="point1_pointpointcoincidentconstraint",
    )
    point2 = models.ForeignKey(
        ReferencePoint,
        on_delete=models.CASCADE,
        related_name="point2_pointpointcoincidentconstraint",
    )

    def residual(self, x1, y1, x2, y2):
        result = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return result

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.point1 == self.point2:
            raise ValidationError(
                "The selected point is already coincident with itself."
            )

        component1 = self.point1.component if self.point1.component else None
        component2 = self.point2.component if self.point2.component else None
        if component1 and component2 and component1 == component2:
            raise ValidationError(
                "The selected points cannot be coincident as they belong to the same component."
            )

    def save(self, *args, **kwargs):
        self.type = "PointPointCoincidentConstraint"
        # Save the instance
        super(PointPointCoincidentConstraint, self).save(*args, **kwargs)


class PointLineCoincidentConstraint(Constraint):
    point = models.ForeignKey(
        ReferencePoint,
        on_delete=models.CASCADE,
        related_name="point_pointlinecoincidentconstraint",
    )
    line = models.ForeignKey(
        ReferenceLine,
        on_delete=models.CASCADE,
        related_name="line_pointlinecoincidentconstraint",
    )

    def residual(self, px, py, lx1, ly1, lx2, ly2):
        # Perpendicular distance from point to line
        numerator = ((ly2 - ly1) * px - (lx2 - lx1) * py + lx2 * ly1 - ly2 * lx1) ** 2
        denominator = (ly2 - ly1) ** 2 + (lx2 - lx1) ** 2
        result = numerator / denominator
        return result

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        self.type = "PointLineCoincidentConstraint"
        # Save the instance
        super(PointLineCoincidentConstraint, self).save(*args, **kwargs)


class DistanceConstraint(Constraint):
    line = models.ForeignKey(
        ReferenceLine, on_delete=models.CASCADE, related_name="line_distanceconstraint"
    )
    value = models.FloatField(validators=[MinValueValidator(0.0)])

    def residual(self, x1, y1, x2, y2):
        distance_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
        target_squared = self.value**2
        result = (distance_squared - target_squared) ** 2
        return result

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

    def residual(self, x1, y1, x2, y2, x3, y3, x4, y4):
        angle = jnp.deg2rad(self.value)
        vector1 = jnp.array([x2 - x1, y2 - y1, 1])
        vector2 = jnp.array([x4 - x3, y4 - y3, 1])
        u = dot_product(vector1, vector2)
        v = vector_magnitude(vector1) * vector_magnitude(vector2)
        result = (jnp.cos(angle) - u / v) ** 2
        return result

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


# FixedConstraint (use references, so points, lines and angles are all valid)

# EqualConstraint (use References, so we can use lines or angles. points are invalid.)

# VerticalConstraint (use lines only)

# HorizontalConstraint (use lines only)

# MidpointConstraint (use 1 point and 1 line, makes point coincident and midpoint)

# Symmetry constraint (1 line, 2 references)

# Collinear constraint (3 points)
