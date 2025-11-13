from django.contrib import admin

from .models import Reference, ReferencePoint, ReferenceLine, ReferenceAngle

# Register your models here.


@admin.register(Reference)
class ReferenceAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "type",
        "label",
        "fixed",
    )
    readonly_fields = ("assembly", "type", "label", "fixed")


@admin.register(ReferencePoint)
class ReferencePointAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "type",
        "label",
        "fixed",
        "component",
        "location",
        "x",
        "y",
    )
    readonly_fields = ("type", "fixed", "matrix")


@admin.register(ReferenceLine)
class ReferenceLineAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "point1",
        "point2",
        "label",
        "fixed",
    )
    readonly_fields = ("type", "fixed", "matrix")


@admin.register(ReferenceAngle)
class ReferenceAngleAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "assembly",
        "line1",
        "line2",
        "label",
        "fixed",
    )
    readonly_fields = ("type", "fixed")


class ReferencePointInline(admin.TabularInline):
    model = ReferencePoint
    extra = 0


class ReferenceLineInline(admin.TabularInline):
    model = ReferenceLine
    extra = 0


class ReferenceAngleInline(admin.TabularInline):
    model = ReferenceAngle
    extra = 0
