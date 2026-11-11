"""Domains by name: retroagi/domains/<name>/domain.py exposes DOMAIN (an
interfaces.Domain). The framework's tools take the name on the command line
and know nothing else of the domain.

A domain's parts can have alternatives by name (Domain.parts: for Super
Mario Bros, its dynamics model). The tools take them as ``--use
PART=NAME``."""

import importlib
from typing import Sequence


def load(name: str, use: Sequence[str] = ()):
    """The domain called ``name``, with the alternatives ``use`` names for
    its parts ("part=alternative" each)."""
    domain = importlib.import_module(f"retroagi.domains.{name}.domain").DOMAIN
    return domain.using(**dict(item.split("=", 1) for item in use)) if use else domain


def arguments(parser) -> None:
    """The command-line options that name a domain."""
    parser.add_argument("--domain", default="smb")
    parser.add_argument("--use", nargs="*", default=(), metavar="PART=NAME",
                        help="alternatives for the domain's parts, e.g. dynamics=observed")


def chosen(args):
    """The domain the options of ``arguments`` name."""
    return load(args.domain, args.use)


def used(domain) -> tuple:
    """``domain``'s chosen alternatives as ``load`` takes them."""
    return tuple(f"{part}={name}" for part, name in sorted(domain.chosen.items()))
