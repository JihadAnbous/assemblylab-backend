from django.conf import settings
from django.db import models
from django.core.exceptions import ValidationError

# Create your models here.


class Template(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name}"


POINT_TYPE_CHOICES = [
    ("Geometrical", "Geometrical"),
    ("Observational", "Observational"),
]


class Point(models.Model):
    template = models.ForeignKey(
        Template,
        on_delete=models.CASCADE,
        related_name="template_point",
    )
    label = models.CharField(max_length=32)
    type = models.CharField(
        max_length=13, choices=POINT_TYPE_CHOICES, default="geometrical"
    )
    order = models.PositiveIntegerField(
        help_text="Adjusting the order will affect the component's contour. Match point adjacency with order."
    )

    def __str__(self):
        return f"{self.template}-{self.label}"


class Component(models.Model):
    template = models.ForeignKey(
        Template, on_delete=models.CASCADE, related_name="template_component"
    )
    number = models.CharField(max_length=100, unique=True)
    description = models.CharField(max_length=100, unique=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.template}-{self.description}"

    def clean(self):
        super().clean()
        # Additional validation

    def save(self, *args, **kwargs):
        # Remove all spaces and convert to uppercase
        self.number = self.number.replace(" ", "").upper()
        # Save the instance
        super(Component, self).save(*args, **kwargs)


UNITS_CHOICES = [
    ("mm", "mm"),
    ("inches", "inches"),
]


class Location(models.Model):
    component = models.ForeignKey(
        Component, on_delete=models.CASCADE, related_name="component_pointdata"
    )
    point = models.ForeignKey(
        Point,
        on_delete=models.CASCADE,
        related_name="point_location",
    )
    x = models.FloatField(help_text="[mm]")
    y = models.FloatField(help_text="[mm]")
    units = models.CharField(max_length=6, choices=UNITS_CHOICES, default="mm")

    def __str__(self):
        return f"{self.point}"

    def clean(self):
        super().clean()
        # Additional validation
        if self.component.template != self.point.template:
            raise ValidationError("This point does not belong to this component.")
