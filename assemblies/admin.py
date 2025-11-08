from django.contrib import admin

from .models import Assembly, AssemblyComponent, AssemblyPoint


# Register your models here.


class AssemblyComponentInline(admin.TabularInline):
    model = AssemblyComponent
    extra = 0


@admin.register(Assembly)
class AssemblyAdmin(admin.ModelAdmin):
    inlines = [AssemblyComponentInline]
    list_display = (
        "id",
        "name",
        "created_by",
        "created_at",
    )
    readonly_fields = ("solved",)


class AssemblyPointInline(admin.TabularInline):
    model = AssemblyPoint
    extra = 0


@admin.register(AssemblyComponent)
class AssemblyComponentAdmin(admin.ModelAdmin):
    inlines = [AssemblyPointInline]
    list_display = (
        "id",
        "assembly",
        "component",
        "fixed",
        "translation_point",
        "pivot_point",
        "x_translation",
        "y_translation",
        "rotation",
    )
    readonly_fields = ("matrix",)
