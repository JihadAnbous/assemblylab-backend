from django.contrib import admin

from .models import Assembly, AssemblyComponent
from constraints.admin import (
    CoincidentConstraintInline,
    DistanceConstraintInline,
    AngleConstraintInline,
    ParallelConstraintInline,
    PerpendicularConstraintInline,
    VariableDistanceConstraintInline,
    VariableAngleConstraintInline,
)
from references.admin import (
    ReferencePointInline,
    ReferenceComponentPointInline,
    ReferenceLineInline,
    ReferenceAngleInline,
)


# Register your models here.


class AssemblyComponentInline(admin.TabularInline):
    model = AssemblyComponent
    extra = 0


@admin.register(Assembly)
class AssemblyAdmin(admin.ModelAdmin):
    inlines = [
        AssemblyComponentInline,
        ReferencePointInline,
        ReferenceLineInline,
        ReferenceAngleInline,
        CoincidentConstraintInline,
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
    readonly_fields = (
        "status",
        "solved",
    )


@admin.register(AssemblyComponent)
class AssemblyComponentAdmin(admin.ModelAdmin):
    inlines = [ReferenceComponentPointInline]
    list_display = (
        "id",
        "assembly",
        "component",
        "fixed",
    )
