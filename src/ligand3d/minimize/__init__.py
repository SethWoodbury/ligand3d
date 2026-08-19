"""Minimization backends."""

from __future__ import annotations

from .base import (
    Availability,
    Backend,
    BackendKind,
    Capabilities,
    MinimizeJob,
    MinimizeResult,
    all_backends,
    check_compatible,
    get_backend,
    list_backends,
    load_builtin_backends,
    parse_chain,
    register,
    resolve_name,
)

load_builtin_backends()

__all__ = [
    "Availability",
    "Backend",
    "BackendKind",
    "Capabilities",
    "MinimizeJob",
    "MinimizeResult",
    "all_backends",
    "check_compatible",
    "get_backend",
    "list_backends",
    "load_builtin_backends",
    "parse_chain",
    "register",
    "resolve_name",
]
