import jax
import jax.numpy as jnp

from .models import ReferenceComponent, ReferenceComponentPoint
from django.db.models.signals import post_save


def create_reference_component_points(sender, instance, created, **kwargs):
    # Once an ReferenceComponent instance is created
    if created:
        # Loop through its Location instances
        for location in instance.component.component_location.all():
            ReferenceComponentPoint.objects.create(
                assembly=instance.assembly,
                type="ReferenceComponentPoint",
                label="Placeholder",
                status="Random",
                reference_component=instance,
                location=location,
                x_plot=location.x,
                y_plot=location.y,
            )


post_save.connect(create_reference_component_points, sender=ReferenceComponent)
