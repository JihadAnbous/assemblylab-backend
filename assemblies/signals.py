import jax
import jax.numpy as jnp


from .models import AssemblyComponent, AssemblyPoint
from django.db.models.signals import post_save

from django_project.utils.helpers import transform


def create_assembly_points(sender, instance, created, **kwargs):
    # Once an AssemblyComponent instance is created
    if created:
        # Loop through its Location instances
        for location in instance.component.component_location.all():
            matrix = location.matrix
            Sx = instance.x_translation
            Sy = instance.y_translation
            rotation = jnp.deg2rad(instance.rotation)
            # Transform the coordinates
            new_coordinates = transform(
                matrix,
                Sx,
                Sy,
                rotation,
            )
            x = float(new_coordinates[0])
            y = float(new_coordinates[1])
            AssemblyPoint.objects.create(
                component=instance, location=location, label="placeholder", x=x, y=y
            )


post_save.connect(create_assembly_points, sender=AssemblyComponent)
