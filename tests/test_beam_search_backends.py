"""Backend dispatch without native dependencies or GPU initialization."""

import pytest

from cayleypy import (
    CayleyGraph,
    PermutationGroups,
    get_beam_search_backend,
    get_default_beam_search_backend,
    register_beam_search_backend,
    set_default_beam_search_backend,
    unregister_beam_search_backend,
)


@pytest.fixture
def graph():
    return CayleyGraph(PermutationGroups.lrx(5), device="cpu", random_seed=42)


@pytest.fixture
def registered():
    previous = get_default_beam_search_backend()
    names = []

    def register(name, callback):
        register_beam_search_backend(name, callback)
        names.append(name)

    yield register
    set_default_beam_search_backend(previous)
    for name in reversed(names):
        unregister_beam_search_backend(name)


def test_default_and_explicit_torch_keep_original_search(graph):
    assert get_default_beam_search_backend() == "torch"
    for options in ({}, {"backend": "torch"}):
        result = graph.beam_search(start_state=[1, 0, 2, 3, 4], return_path=True, **options)
        assert result.path == [2]
        assert graph.apply_path([1, 0, 2, 3, 4], result.path).reshape(-1).tolist() == [0, 1, 2, 3, 4]


def test_callable_receives_identical_objects(graph):
    sentinel, state, predictor = object(), object(), object()
    calls = []

    def custom(received_graph, **kwargs):
        calls.append((received_graph, kwargs))
        return sentinel

    assert graph.beam_search(backend=custom, start_state=state, predictor=predictor) is sentinel
    assert calls == [(graph, {"start_state": state, "predictor": predictor})]
    assert get_default_beam_search_backend() == "torch"


def test_registered_default_explicit_override_and_restore(graph, registered):
    sentinel = object()

    def custom(_graph, **_kwargs):
        return sentinel

    registered("test_external", custom)
    assert get_beam_search_backend("test_external") is custom
    assert graph.beam_search(backend="test_external") is sentinel
    assert set_default_beam_search_backend("test_external") == "torch"
    assert get_default_beam_search_backend() == "test_external"
    assert get_beam_search_backend() is custom
    assert graph.beam_search() is sentinel
    assert graph.beam_search(backend="torch", start_state=[1, 0, 2, 3, 4], return_path=True).path == [2]
    assert set_default_beam_search_backend("torch") == "test_external"


def test_backend_can_fallback_without_recursive_dispatch(graph, registered):
    def custom(received_graph, **kwargs):
        return get_beam_search_backend("torch")(received_graph, **kwargs)

    registered("test_fallback", custom)
    set_default_beam_search_backend("test_fallback")
    assert graph.beam_search(start_state=[1, 0, 2, 3, 4], return_path=True).path == [2]


def test_backend_error_is_not_swallowed(graph):
    error = RuntimeError("worker failed")

    def failing(_graph, **_kwargs):
        raise error

    with pytest.raises(RuntimeError) as caught:
        graph.beam_search(backend=failing)
    assert caught.value is error


def test_registry_rejects_collisions_and_protects_default(registered):
    registered("test_owned", lambda _graph, **_kwargs: None)
    for name in ("torch", "test_owned"):
        with pytest.raises(ValueError, match="already registered"):
            register_beam_search_backend(name, lambda _graph: None)
    set_default_beam_search_backend("test_owned")
    for name in ("torch", "test_owned"):
        with pytest.raises(ValueError, match="cannot unregister"):
            unregister_beam_search_backend(name)


@pytest.mark.parametrize("selector", ["missing_backend", 17, object()])
def test_invalid_selection_has_no_silent_fallback(graph, selector):
    with pytest.raises((ValueError, TypeError)):
        graph.beam_search(backend=selector)


@pytest.mark.parametrize("name", ["", " x", "x ", None, 17])
def test_invalid_registration_name(name):
    with pytest.raises(ValueError):
        register_beam_search_backend(name, lambda _graph: None)


def test_invalid_callback_and_default_do_not_mutate_registry():
    with pytest.raises(TypeError):
        register_beam_search_backend("test_bad", None)
    with pytest.raises(ValueError):
        set_default_beam_search_backend("missing_backend")
    with pytest.raises(TypeError):
        set_default_beam_search_backend(None)
    assert get_default_beam_search_backend() == "torch"
