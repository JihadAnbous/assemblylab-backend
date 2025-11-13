from django.core.exceptions import ValidationError

from django.db import models
from django.apps import apps

from components.models import Location
from assemblies.models import Assembly, AssemblyComponent

# Create your models here.


class Reference(models.Model):
    assembly = models.ForeignKey(
        Assembly, on_delete=models.CASCADE, related_name="assembly_reference"
    )
    type = models.CharField(max_length=50, editable=False)
    label = models.CharField(max_length=100, blank=True, null=True)
    fixed = models.BooleanField(default=False, editable=False)

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


class ReferencePoint(Reference):
    component = models.ForeignKey(
        AssemblyComponent,
        on_delete=models.CASCADE,
        related_name="component_referencepoint",
    )
    location = models.ForeignKey(
        Location, on_delete=models.CASCADE, related_name="location_referencepoint"
    )
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

    def __str__(self):
        return f"{self.assembly}-{self.label}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.point1.component.assembly != self.assembly:
            raise ValidationError("Point 1 does not belong to this assembly.")
        if self.point2.component.assembly != self.assembly:
            raise ValidationError("Point 2 does not belong to this assembly.")
        if self.point1 == self.point2:
            raise ValidationError("You cannot make a line out of the same point.")

    def save(self, *args, **kwargs):
        if self.point1.fixed and self.point2.fixed is True:
            self.fixed = True
        if self.fixed is True:
            # Create a variable that stores line colour? Bold = Fixed/constrained ??
            pass
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
