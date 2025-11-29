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
        "status",
        "hidden",
    )
    readonly_fields = ("assembly", "type", "label", "status")


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
        "status",
        "component",
        "hidden",
    )
    readonly_fields = ("type", "status")


@admin.register(ReferencePoint)
class ReferencePointAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "type",
        "label",
        "status",
        "component",
        "location",
        "x_plot",
        "y_plot",
        "hidden",
    )
    readonly_fields = ("type", "status", "matrix")


@admin.register(ReferenceLine)
class ReferenceLineAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "point1",
        "point2",
        "label",
        "status",
        "hidden",
    )
    readonly_fields = ("type", "status", "matrix")


@admin.register(ReferenceAngle)
class ReferenceAngleAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "line1",
        "line2",
        "label",
        "status",
        "hidden",
    )
    readonly_fields = ("type", "status")


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
