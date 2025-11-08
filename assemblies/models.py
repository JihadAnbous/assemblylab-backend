from django.db import models

from components.models import Component, Location

# Create your models here.


class Assembly(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return f"{self.name}"


class AssemblyComponent(models.Model):
    assembly = models.ForeignKey(
        Assembly, on_delete=models.CASCADE, related_name="assembly_assemblycomponent"
    )
    component = models.ForeignKey(
        Component, on_delete=models.CASCADE, related_name="component_assemblycomponent"
    )

    def __str__(self):
        return f"{self.assembly}-{self.component}"
