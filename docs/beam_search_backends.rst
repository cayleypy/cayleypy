Optional beam search backends
=============================

Existing ``graph.beam_search(...)`` calls use the builtin PyTorch algorithm until
an application explicitly selects another backend. Installing a package never
activates it: CayleyPy does not discover plugins, download sources, compile CUDA
or load a native library at import time.

An external backend receives the original graph and search keyword arguments:

.. code-block:: python

   result = graph.beam_search(backend=my_backend, start_state=start, predictor=predictor)

``my_backend(graph, **kwargs)`` should return a ``BeamSearchResult`` (or subclass)
with paths in the original graph's generator order. It owns capability checks,
execution, path validation and any fallback policy. CayleyPy forwards arguments,
results and errors unchanged. It does not promise identical search frontiers,
path lengths, scoring or performance across backends.

Packages can provide an explicit enable/disable function using the registry:

.. code-block:: python

   from cayleypy import (
       register_beam_search_backend, set_default_beam_search_backend,
       unregister_beam_search_backend,
   )

   register_beam_search_backend("my_backend", my_backend)
   previous = set_default_beam_search_backend("my_backend")
   try:
       result = graph.beam_search(start_state=start, predictor=predictor)
   finally:
       set_default_beam_search_backend(previous)
       unregister_beam_search_backend("my_backend")

Configure the process default before launching application threads. Registry
updates are locked; running backend calls execute outside the lock. Calls
resolve the default when invoked, including methods saved before a default
change. Prefer a per-call callable when independent consumers need separate
configuration. Duplicate names are rejected. Restore the default before
unregistering it; the builtin ``torch`` cannot be replaced or removed.

``graph.beam_search(backend="torch", ...)`` always bypasses external backends.
It accepts the original Torch predictor and search options, not plugin-specific
objects/options. An external backend implementing fallback can resolve the
builtin with ``get_beam_search_backend("torch")`` and call it directly without
recursing through the process default. No exception triggers automatic fallback
inside CayleyPy.

The optional `MultiGPUBeamSearch adapter
<https://github.com/TryDotAtwo/MultiGPUBeamSearch/tree/main/integrations/cayleypy_native>`_
uses this hook for CUDA/NCCL beam search, with its own installation requirements,
supported graph/model contracts and explicit fallback policy. CayleyPy does not
depend on that package or its CUDA toolchain.
