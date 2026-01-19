from django.contrib import admin

from .models import (
    Reference,
    ReferencePoint,
    ReferenceComponent,
    ReferenceLine,
    ReferenceAngle,
)

# Register your models here.


@admin.register(Reference)
class ReferenceAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "type",
        "label",
    )
    readonly_fields = (
        "assembly",
        "type",
        "label",
    )


class ReferencePointInline(admin.TabularInline):
    model = ReferencePoint
    fk_name = "component"
    extra = 0


@admin.register(ReferenceComponent)
class ReferenceComponentAdmin(admin.ModelAdmin):
    inlines = [ReferencePointInline]
    list_display = (
        "id",
        "assembly",
        "type",
        "label",
        "component",
    )
    readonly_fields = ("type", "dof")


@admin.register(ReferencePoint)
class ReferencePointAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "type",
        "label",
        "component",
        "location",
        "x_plot",
        "y_plot",
    )
    readonly_fields = ("type", "matrix", "dof")


@admin.register(ReferenceLine)
class ReferenceLineAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "point1",
        "point2",
        "label",
    )
    readonly_fields = (
        "type",
        "matrix",
    )


@admin.register(ReferenceAngle)
class ReferenceAngleAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "line1",
        "line2",
        "label",
    )
    readonly_fields = ("type",)


class ReferencePointInline(admin.TabularInline):
    model = ReferencePoint
    extra = 0


class ReferenceComponentInline(admin.TabularInline):
    model = ReferenceComponent
    extra = 0


class ReferenceLineInline(admin.TabularInline):
    model = ReferenceLine
    extra = 0


class ReferenceAngleInline(admin.TabularInline):
    model = ReferenceAngle
    extra = 0
