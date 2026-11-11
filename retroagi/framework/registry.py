"""Domains by name: retroagi/domains/<name>/domain.py exposes DOMAIN (an
interfaces.Domain). The framework's tools take the name on the command line
and know nothing else of the domain."""

import importlib


def load(name: str):
    """The domain called ``name``."""
    return importlib.import_module(f"retroagi.domains.{name}.domain").DOMAIN
