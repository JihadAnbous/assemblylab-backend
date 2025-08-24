from django.contrib import admin

from .models import (
    ComponentType,
    Component,
    ComponentPoint,
)

# Register your models here.


@admin.register(ComponentType)
class ComponentTypeAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
    )


class ComponentPointInline(admin.TabularInline):
    model = ComponentPoint
    extra = 0


@admin.register(ComponentPoint)
class ComponentPointAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "x",
        "y",
    )


@admin.register(Component)
class ComponentAdmin(admin.ModelAdmin):
    inlines = [
        ComponentPointInline,
    ]
    list_display = (
        "id",
        "type",
        "description",
    )
