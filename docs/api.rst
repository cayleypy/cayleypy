CayleyPy API Reference
======================


Core classes and functions
''''''''''''''''''''''''''


.. toctree::
   :maxdepth: 1

.. autosummary::
    :toctree: generated/

    cayleypy.CayleyGraphDef
    cayleypy.CayleyGraph
    cayleypy.CayleyPath
    cayleypy.find_path

Graphs library
''''''''''''''


.. autosummary::
    :toctree: generated/

    cayleypy.PermutationGroups
    cayleypy.MatrixGroups
    cayleypy.Puzzles
    cayleypy.GapPuzzles
    cayleypy.create_graph
    cayleypy.prepare_graph

Beam search and ML
''''''''''''''''''


.. autosummary::
    :toctree: generated/

    cayleypy.Predictor
    cayleypy.algo.BeamSearchAlgorithm
    cayleypy.algo.BeamSearchResult
    cayleypy.algo.RandomWalksGenerator
    cayleypy.models.ModelConfig

BFS algorithm and its variations
''''''''''''''''''''''''''''''''


.. autosummary::
    :toctree: generated/

    cayleypy.algo.BfsAlgorithm
    cayleypy.algo.BfsResult
    cayleypy.algo.bfs_bitmask
    cayleypy.algo.bfs_numpy
    cayleypy.algo.InteractiveBfs
    cayleypy.algo.MeetInTheMiddle
