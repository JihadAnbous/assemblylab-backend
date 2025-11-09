from django.contrib import admin

from .models import (
    Constraint,
    CoincidentConstraint,
    DistanceConstraint,
    AngleConstraint,
)

# Register your models here.


@admin.register(Constraint)
class ConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type")
    readonly_fields = (
        "assembly",
        "type",
    )


@admin.register(CoincidentConstraint)
class CoincidentConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "point1", "point2")
    readonly_fields = ("type",)


@admin.register(DistanceConstraint)
class DistanceConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "line", "value")
    readonly_fields = ("type",)


@admin.register(AngleConstraint)
class AngleConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "angle", "value")
    readonly_fields = ("type",)
