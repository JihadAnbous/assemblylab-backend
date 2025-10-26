from django.contrib import admin

from .models import ComponentType, Component, ComponentPoint, ComponentCoordinate

# Register your models here.


class ComponentCoordinateInline(admin.TabularInline):
    model = ComponentCoordinate
    extra = 0


class ComponentPointInline(admin.TabularInline):
    model = ComponentPoint
    extra = 0


@admin.register(ComponentType)
class ComponentTypeAdmin(admin.ModelAdmin):
    inlines = [ComponentPointInline]
    list_display = (
        "id",
        "name",
    )


@admin.register(Component)
class ComponentAdmin(admin.ModelAdmin):
    inlines = [ComponentCoordinateInline]
    list_display = (
        "id",
        "type",
        "name",
    )
