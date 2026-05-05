"""Re-export shim: exposes MjolnirResearch under the agamotto namespace.

The canonical implementation lives in mjolnir/core/research.py. This module
allows downstream scripts and tests to import it as::

    from agamotto.research_mjolnir import MjolnirResearch
"""
from mjolnir.core.research import MjolnirResearch  # noqa: F401

__all__ = ["MjolnirResearch"]
