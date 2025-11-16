import jax
import jax.numpy as jnp

from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator

from django.db import models
from django.apps import apps

from assemblies.models import Assembly
from references.models import Reference, ReferencePoint, ReferenceLine, ReferenceAngle


from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType

from django_project.utils.constraints import coincident_constraint

# Create your models here.


class Constraint(models.Model):
    assembly = models.ForeignKey(
        Assembly, on_delete=models.CASCADE, related_name="assembly_constraint"
    )
    type = models.CharField(max_length=50, editable=False)
    is_explicit = models.BooleanField(default=False, editable=False)

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


class CoincidentConstraint(Constraint):
    reference1 = models.ForeignKey(
        Reference,
        on_delete=models.CASCADE,
        related_name="reference1_coincidentconstraint",
    )
    reference2 = models.ForeignKey(
        Reference,
        on_delete=models.CASCADE,
        related_name="reference2_coincidentconstraint",
    )

    @property
    def residual(self):
        if self.is_explicit:
            # send signal to update the other reference
            # OR
            # ref2 = self.reference2.get_specific_instance()
            # ref2.x = ref1.x       ref2.y = ref1.y
            pass
        else:
            # Construct the residual equation
            result = None
            ref1 = self.reference1.get_specific_instance()
            ref2 = self.reference2.get_specific_instance()

            # If ref1.fixed == True:
            # ref1_matrix = ref1.matrix
            # Else:
            # ref1_matrix = jnp.array([x1, y1, 1])

            # If 2 points were selected
            # if ref1.type == "ReferencePoint" and ref2.type == "ReferencePoint":
            # result = coincident_constraint(ref1_matrix, ref2_matrix)
            # If 1 point and 1 line was selected
            # If 2 lines were selected
        return result

    # @property
    # def derivative(self):
    #     # if is_explicit, return None. For jacobian, if implicit, include, otherwise don't.
    #     # If is_explicit == false, then derive the residual
    #     result = jax.grad(self.equation)
    #     return result

    def __str__(self):
        return f"{self.type}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.reference1 == self.reference2:
            raise ValidationError(
                "The selected reference is already coincident with itself."
            )

    def save(self, *args, **kwargs):
        # Set constraint to explicit if a reference is fixed
        if self.reference1.fixed or self.reference2.fixed:
            self.is_explicit = True
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


# FixedConstraint (use references, so points, lines and angles are all valid)

# EqualConstraint (use References, so we can use lines or angles. points are invalid.)

# VerticalConstraint (use lines only)

# HorizontalConstraint (use lines only)

# MidpointConstraint (use 1 point and 1 line, makes point coincident and midpoint)

# Symmetry constraint (1 line, 2 references)

# Collinear constraint (3 points)
