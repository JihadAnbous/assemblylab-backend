from django.contrib import admin

from .models import (
    Constraint,
    CoincidentConstraint,
    DistanceConstraint,
    AngleConstraint,
    ParallelConstraint,
    PerpendicularConstraint,
    VariableDistanceConstraint,
    VariableAngleConstraint,
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


@admin.register(ParallelConstraint)
class ParallelConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "line1", "line2")
    readonly_fields = ("type",)


@admin.register(PerpendicularConstraint)
class PerpendicularConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "line1", "line2")
    readonly_fields = ("type",)


@admin.register(VariableDistanceConstraint)
class VariableDistanceConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "line", "minimum", "maximum")
    readonly_fields = ("type",)


@admin.register(VariableAngleConstraint)
class VariableAngleConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "angle", "minimum", "maximum")
    readonly_fields = ("type",)


class CoincidentConstraintInline(admin.TabularInline):
    model = CoincidentConstraint
    extra = 0


class DistanceConstraintInline(admin.TabularInline):
    model = DistanceConstraint
    extra = 0


class AngleConstraintInline(admin.TabularInline):
    model = AngleConstraint
    extra = 0


class ParallelConstraintInline(admin.TabularInline):
    model = ParallelConstraint
    extra = 0


class PerpendicularConstraintInline(admin.TabularInline):
    model = PerpendicularConstraint
    extra = 0


class VariableDistanceConstraintInline(admin.TabularInline):
    model = VariableDistanceConstraint
    extra = 0


class VariableAngleConstraintInline(admin.TabularInline):
    model = VariableAngleConstraint
    extra = 0
