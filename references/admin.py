from django.contrib import admin

from .models import Reference, ReferenceLine, ReferenceAngle

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
    read_only = ("assembly", "type", "label", "fixed")


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
    read_only = ("type", "fixed")


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
    read_only = ("type", "fixed")


class ReferenceLineInline(admin.TabularInline):
    model = ReferenceLine
    extra = 0


class ReferenceAngleInline(admin.TabularInline):
    model = ReferenceAngle
    extra = 0
