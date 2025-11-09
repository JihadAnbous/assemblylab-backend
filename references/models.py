from django.db import models
from django.core.exceptions import ValidationError

from assemblies.models import Assembly, AssemblyPoint

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


class ReferenceLine(Reference):
    point1 = models.ForeignKey(
        AssemblyPoint, on_delete=models.CASCADE, related_name="point1_referenceline"
    )
    point2 = models.ForeignKey(
        AssemblyPoint, on_delete=models.CASCADE, related_name="point2_referenceline"
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
            # Try to fix self.x and self.y to the assembly component's matrix - packing/unpacking functions
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
        if self.fixed is True:
            # Try to fix self.x and self.y to the assembly component's matrix - packing/unpacking functions
            pass
        self.type = "ReferenceAngle"
        # Save the instance
        super(ReferenceAngle, self).save(*args, **kwargs)
