from .models import FixedPointConstraint
from references.models import ReferencePoint
from django.db.models.signals import post_save


def fix_reference_point(sender, instance, created, **kwargs):
    # Once an FixedPointConstraint is created
    if created:
        # Update the "fixed" field of the reference point
        instance.point.fixed = True
        instance.point.save()


post_save.connect(fix_reference_point, sender=FixedPointConstraint)
