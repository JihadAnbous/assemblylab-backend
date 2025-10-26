from django.db import models
from django.core.exceptions import ValidationError

# Create your models here.


class ComponentType(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class Component(models.Model):
    type = models.ForeignKey(
        ComponentType, on_delete=models.CASCADE, related_name="type_component"
    )
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return f"{self.type} - {self.name}"


class ComponentPoint(models.Model):
    type = models.ForeignKey(
        ComponentType, on_delete=models.CASCADE, related_name="type_componentpoint"
    )
    name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.type}-{self.name}"


class ComponentCoordinate(models.Model):
    component = models.ForeignKey(
        Component,
        on_delete=models.CASCADE,
        related_name="component_componentcoordinate",
    )
    point = models.ForeignKey(
        ComponentPoint,
        on_delete=models.CASCADE,
        related_name="point_componentcoordinate",
    )
    x = models.FloatField(help_text="[mm]")
    y = models.FloatField(help_text="[mm]")

    def __str__(self):
        return f"{self.x}, {self.y}"

    def clean(self):
        super().clean()
        if self.component and self.point:
            if self.component.type != self.point.type:
                raise ValidationError(
                    f"Point '{self.point}' does not belong to component type '{self.component.type}'."
                )
            # Add another to remove duplicates
