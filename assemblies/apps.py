from django.apps import AppConfig


class AssembliesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "assemblies"

    def ready(self):
        import assemblies.signals  # This imports and registers your signals
