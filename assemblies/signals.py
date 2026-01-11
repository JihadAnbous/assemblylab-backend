from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Assembly


# @receiver(post_save, sender=Assembly)
# def solve_on_assembly_change(sender, instance, **kwargs):
#     instance.solve_geometry()
