#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import yaml
sys.path.append(".")
from summary_app.inference import SummaryExtractor
from summary_app.schema.config import AppConfig



CONFIG_PATH = "configs/app.yaml"


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

    config_json = {}
    with open(CONFIG_PATH, "r") as f:
        config_json = yaml.load(f, yaml.SafeLoader)
    configs = AppConfig(**config_json)
    SummaryExtractor.init_instance(app_config=configs)

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(['manage.py', "runserver"])


if __name__ == '__main__':
    main()
