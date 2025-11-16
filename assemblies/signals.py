import jax
import jax.numpy as jnp

from references.models import ReferenceComponentPoint
from .models import AssemblyComponent
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
            ReferenceComponentPoint.objects.create(
                assembly=instance.assembly,
                type="ReferenceComponentPoint",
                label="placeholder",
                status="Fixed",
                component=instance,
                location=location,
                x_plot=x,
                y_plot=y,
            )


post_save.connect(create_assembly_points, sender=AssemblyComponent)
