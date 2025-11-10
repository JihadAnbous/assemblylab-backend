from .models import AssemblyComponent, AssemblyPoint
from django.db.models.signals import post_save


def create_assembly_points(sender, instance, created, **kwargs):
    if created:  # Only run when a new AssemblyComponent is created
        for location in instance.component.component_location.all():
            AssemblyPoint.objects.create(
                component=instance, location=location, label="x"
            )


post_save.connect(create_assembly_points, sender=AssemblyComponent)
