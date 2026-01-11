from django.contrib import admin

from .models import Assembly
from constraints.admin import (
    FixedPointConstraintInline,
    PointPointCoincidentConstraintInline,
    PointLineCoincidentConstraintInline,
    DistanceConstraintInline,
    AngleConstraintInline,
    ParallelConstraintInline,
    PerpendicularConstraintInline,
    VariableDistanceConstraintInline,
    VariableAngleConstraintInline,
)
from references.admin import (
    ReferencePointInline,
    ReferenceComponentInline,
    ReferenceLineInline,
    ReferenceAngleInline,
)


# Register your models here.


@admin.register(Assembly)
class AssemblyAdmin(admin.ModelAdmin):
    inlines = [
        ReferenceComponentInline,
        ReferencePointInline,
        ReferenceLineInline,
        ReferenceAngleInline,
        FixedPointConstraintInline,
        PointPointCoincidentConstraintInline,
        PointLineCoincidentConstraintInline,
        DistanceConstraintInline,
        AngleConstraintInline,
        ParallelConstraintInline,
        PerpendicularConstraintInline,
        VariableDistanceConstraintInline,
        VariableAngleConstraintInline,
    ]
    list_display = (
        "id",
        "name",
        "created_by",
        "created_at",
    )
    readonly_fields = ("status", "solved", "point_map", "constraints")
