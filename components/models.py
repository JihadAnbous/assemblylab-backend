from django.db import models

# Create your models here.


class ComponentType(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class Component(models.Model):
    type = models.ForeignKey(
        ComponentType, on_delete=models.CASCADE, related_name="type_componenttype"
    )
    description = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return f"{self.type} - {self.description}"


class ComponentPoint(models.Model):
    component = models.ForeignKey(
        Component, on_delete=models.CASCADE, related_name="component_component"
    )
    name = models.CharField(max_length=100, unique=True)
    x = models.FloatField(help_text="[mm]")
    y = models.FloatField(help_text="[mm]")

    def __str__(self):
        return f"{self.name}: {self.x}, {self.y}"
