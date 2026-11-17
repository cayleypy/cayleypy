"""Explicit, optional whole-search backends for :meth:`CayleyGraph.beam_search`.

Importing this module does not discover or import third-party plugins. Backends
own their capability checks, fallback policy and result validation.
"""

from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

from .algo.beam_search import BeamSearchAlgorithm

if TYPE_CHECKING:
    from .cayley_graph import CayleyGraph
    from .algo.beam_search_result import BeamSearchResult


class BeamSearchBackend(Protocol):
    """Callable accepting a graph and the caller's search keyword arguments."""

    def __call__(self, graph: "CayleyGraph", **kwargs: Any) -> "BeamSearchResult": ...


def _torch_backend(graph, **kwargs):
    return BeamSearchAlgorithm(graph).search(**kwargs)


_LOCK = RLock()
_BACKENDS: Dict[str, BeamSearchBackend] = {"torch": _torch_backend}
_DEFAULT = "torch"


def register_beam_search_backend(name: str, backend: BeamSearchBackend) -> None:
    """Register an explicit callable. Existing names, including torch, are reserved."""
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("backend name must be a nonempty string without surrounding whitespace")
    if not callable(backend):
        raise TypeError("backend must be callable")
    with _LOCK:
        if name in _BACKENDS:
            raise ValueError(f"beam search backend {name!r} is already registered")
        _BACKENDS[name] = backend


def unregister_beam_search_backend(name: str) -> None:
    """Remove a backend after restoring the default. The builtin torch cannot be removed."""
    with _LOCK:
        if name in ("torch", _DEFAULT):
            raise ValueError("cannot unregister the builtin torch or the current default backend")
        if name not in _BACKENDS:
            raise ValueError(f"unknown beam search backend {name!r}")
        del _BACKENDS[name]


def get_default_beam_search_backend() -> str:
    """Return the process-wide default name (initially torch)."""
    with _LOCK:
        return _DEFAULT


def get_beam_search_backend(name: Optional[str] = None) -> BeamSearchBackend:
    """Resolve a registered name, or the current default when name is None."""
    with _LOCK:
        selected = _DEFAULT if name is None else name
        if not isinstance(selected, str):
            raise TypeError("backend must be a registered name, callable, or None")
        if selected not in _BACKENDS:
            raise ValueError(f"unknown beam search backend {selected!r}; register it explicitly before use")
        return _BACKENDS[selected]


def set_default_beam_search_backend(name: str) -> str:
    """Set the process default and return its previous name. Configure before starting threads."""
    global _DEFAULT  # pylint: disable=global-statement
    with _LOCK:
        get_beam_search_backend(name)
        if name is None:
            raise TypeError("default backend must be a registered name")
        previous, _DEFAULT = _DEFAULT, name
        return previous
