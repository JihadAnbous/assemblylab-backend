from django.contrib import admin

from .models import (
    Constraint,
    FixedPointConstraint,
    PointPointCoincidentConstraint,
    PointLineCoincidentConstraint,
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


@admin.register(FixedPointConstraint)
class FixedPointConstraintAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "type",
        "point",
        "x_value",
        "y_value",
    )
    readonly_fields = ("type", "symbols", "r", "j")


@admin.register(PointPointCoincidentConstraint)
class PointPointCoincidentConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "point1", "point2")
    readonly_fields = ("type", "symbols", "r", "j")


@admin.register(PointLineCoincidentConstraint)
class PointLineCoincidentConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "point", "line")
    readonly_fields = ("type",)


@admin.register(DistanceConstraint)
class DistanceConstraintAdmin(admin.ModelAdmin):
    list_display = ("id", "assembly", "type", "line", "value")
    readonly_fields = ("type", "symbols", "r", "j")


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


class FixedPointConstraintInline(admin.TabularInline):
    model = FixedPointConstraint
    extra = 0


class PointPointCoincidentConstraintInline(admin.TabularInline):
    model = PointPointCoincidentConstraint
    extra = 0


class PointLineCoincidentConstraintInline(admin.TabularInline):
    model = PointLineCoincidentConstraint
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
