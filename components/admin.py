from django.contrib import admin

from .models import Template, Point, Component, Location

# Register your models here.


class PointInline(admin.TabularInline):
    model = Point
    extra = 0


@admin.register(Template)
class TemplateAdmin(admin.ModelAdmin):
    inlines = [PointInline]
    list_display = (
        "id",
        "name",
        "created_by",
        "created_at",
    )


class LocationInline(admin.TabularInline):
    model = Location
    extra = 0


@admin.register(Component)
class ComponentAdmin(admin.ModelAdmin):
    inlines = [LocationInline]
    list_display = (
        "id",
        "template",
        "number",
        "description",
        "created_by",
        "created_at",
    )
