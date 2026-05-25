from __future__ import annotations

from .openclaw import OPENCLAW_BACKEND
from .base import RuntimeBackend

REGISTERED_BACKENDS: dict[str, RuntimeBackend] = {
    OPENCLAW_BACKEND.name: OPENCLAW_BACKEND,
}
DEFAULT_RUNTIME_BACKEND = OPENCLAW_BACKEND.name


def get_runtime_backend(name: str) -> RuntimeBackend:
    try:
        return REGISTERED_BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(REGISTERED_BACKENDS))
        raise ValueError(f"Unsupported runtime backend: {name}. Available backends: {available}") from exc


def list_runtime_backends() -> tuple[str, ...]:
    return tuple(sorted(REGISTERED_BACKENDS))


__all__ = [
    "DEFAULT_RUNTIME_BACKEND",
    "REGISTERED_BACKENDS",
    "RuntimeBackend",
    "get_runtime_backend",
    "list_runtime_backends",
]