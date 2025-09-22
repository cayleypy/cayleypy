# CayleyPy

AI-based libarary to work with googol-size graphs.
Supporting:  Cayley graphs, Schreier coset graphs, more to be added.


## Overview

Exteremely large graphs (e.g. googol size) cannot be approached in a usual way,
it is impossible neither to create, neither to store them by standard methods.

Typically such graphs arise as state-transition graphs.
For chess, Go or any other games - nodes of the graphs are positions, edges correspond to moves between them.
For Rubik's cube - nodes are configurations, edges corresponds to configurations different by single moves. 

The most simple and clear examples of such graphs - are [Caley graphs](https://en.wikipedia.org/wiki/Cayley_graph) in mathematics.
(and [Schreier coset graphs](https://en.wikipedia.org/wiki/Schreier_coset_graph) ). 
Initial developments will focus on these graphs, supporting other types later. 

We plan to support:

* ML/RL methods for pathfinding 
* Estimation of diameters and growths
* Embeddings
* Efficient BFS for small subgraphs
* Efficient random walks generation
* Efficient Beam Search 
* Hamiltionan paths finding
* Efficient computing on CPU, GPU, TPU (with JAX), usable on Kaggle.
* Etc. 

Mathematical applications: 
* Estimation of diameters and growths
* Approximation of the word metrics and diffusion distnace
* Estimation of the mixing time for random walks of different types 
* BFS from given state (growth function, adjacency matrix, last layers).
* Library of graphs and generators (LRX, TopSpin, Rubik Cubes, wreath, globe etc.,
  see [here](https://www.kaggle.com/code/ivankolt/generation-of-incidence-mtx-pancake)).
* Library of datasets with solutions to some problems (e.g. growth functions like
  [here](https://www.kaggle.com/code/fedimser/bfs-for-binary-string-permutations)).

## Examples

See the following Kaggle notebooks for examples of library usage:

* [Basic usage](https://www.kaggle.com/code/fedimser/cayleypy-demo) - defining Cayley graphs for permutation and matrix groups, running BFS, getting explicit Networkx graphs.
* [Computing spectra](https://www.kaggle.com/code/fedimser/computing-spectra-of-cayley-graphs-using-cayleypy).
* [Library of puzzles in GAP format in CayleyPy](https://www.kaggle.com/code/fedimser/library-of-puzzles-in-gap-format-in-cayleypy).
* Path finding in Cayley Graphs:
  * [Beam seacrh with CayleyPy](https://www.kaggle.com/code/fedimser/beam-search-with-cayleypy) - simple example of finding paths for LRX (n=12) using beam search and neural network.
  * [Finidng shortest paths for LRX (n=8) using BFS](https://www.kaggle.com/code/fedimser/lrx-solution).
  * [Finidng shortest paths for LRX cosets (n=16 and n=32) using BFS](https://www.kaggle.com/code/fedimser/lrx-binary-with-cayleypy-bfs-only).
  * [Beam search with neural network for LRX cosets (n=32)](https://www.kaggle.com/code/fedimser/solve-lrx-binary-with-cayleypy).
  * [Beam search for LRX, n=16](https://www.kaggle.com/code/fedimser/lrx-solution-n-16-beamsearch). 
  * [Beam search for LRX, n=32](https://www.kaggle.com/code/fedimser/lrx-solution-n-32-beamsearch)
* Growth function computations:
  * [For LX](https://www.kaggle.com/code/fedimser/growth-function-for-lx-cayley-graph).
  * [For TopSpin cosets](https://www.kaggle.com/code/fedimser/growth-functions-for-topspin-cosets).
* Becnhmarks:
  * [Benchmarks versions of BFS in CayleyPy](https://www.kaggle.com/code/fedimser/benchmark-versions-of-bfs-in-cayleypy).
  * [Becnhmark BFS on GPU](https://www.kaggle.com/code/fedimser/benchmark-bfs-in-cayleypy-on-gpu-p100).

## Installation

We recommend installing the latest version from GitHub:

```
pip install git+https://github.com/cayleypy/cayleypy
```

You may also install using pip, although this might be missing recently added features:

```
pip install cayleypy
```

## Documentation

Documentation (API reference) for the latest version of the library is available
[here](https://cayleypy.github.io/cayleypy-docs/api.html).

## Development

To start development, run:

```
git clone https://github.com/cayleypy/cayleypy.git
cd cayleypy
pip install -e .[torch,lint,test,dev,docs]
```

To run all tests, including some slow running tests:

```
RUN_SLOW_TESTS=1 pytest
```

Before commiting, run these checks:

```
./lint.sh
pytest 
```

To check coverage, run:

```
coverage run -m pytest && coverage html
```

To rebuild documentation locally, run:

```
./docs/build_docs.sh 
```

### Formatting

This repository uses the [Black formatter](https://github.com/psf/black).
If you are getting error saying that some files "would be reformatted", you need to format
your code using Black. There are few convenient ways to do that:
* From command line: run `black .` 
* In PyCharm: go to Setting>Tools>Black, and check "Use Black formatter": "On code reformat" 
    (then it will run on Ctrl+Alt+L), or "On save", or both.
* In Visual Studio code: install the
    [Black Formatter extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter),
    then use Ctrl+Shift+I to format code. 
    If you are  asked to configure default formatter, pick the Black formatter.

### Style

* In general, this repository follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). All
    contrbutors should read it.
* When writing comments, [use punctuation](https://google.github.io/styleguide/pyguide.html#386-punctuation-spelling-and-grammar).
    In particular, always put a period (".") in the end of sentences.
* We have pylint checks to enforce some style rules. You should fix pylint warnings instead of disabling the check. 

## How to add a new Cayley graph

Cayley graphs must be defined by a function that returns `CayleyGraphDef`. 
First, you need to decide where in the library to put it:
* If it's a graph generated by permutations, the function should be added to  
    `PermutationGroups` in `cayleypy/graphs_lib.py`, annotated as `@staticmethod`.
* If it's a graph generated by matrices, the function should be added to  
    `MatrixGroups` in `cayleypy/graphs_lib.py`.
* If it's a graph for a physical puzzle, the function should be added to 
    `Puzzles` in `caylepy/puzzles/puzzles.py`. If it requires non-trivial construction,
    move that to separate function(s) and put them in separate file in `cayleypy/puzzles`.
    If the puzzle is defined by hardcoded permutations, put them in `cayleypy/puzzles/moves.py`. 
* If it's a graph for a puzzle, and you have definition in GAP format, put the `.gap` file in
    `puzzles/gap_files/default`. It will become available via `cayleypy.GapPuzzles`.
* If it's a new type of graph, check with @fedimser where to put it.

Do not add new graphs to `prepare_graph`! We want new graphs to be added in different 
places to avoid merge conflicts.

Then, you need to define your graph. Definition consists of the following:
* Generators.
* Generator names (optional).
* Central state (optional, defaults to neutral element in the group, e.g. 
    identity permutation).

When you are ready, do the following:
1. Create a new branch in this repository (not a fork).
2. Add your function where you decided. See how other graphs are defined and follow that as an example.
3. Write a docstring for your function, describing your graph. If possible, include reference
   (e.g. to Wikipedia article, Arxiv paper or a book) where the graph is defined.
4. Add a test that creates an instance of your graph for small size and checks something about it 
     (at least check number of generators).
5. Create a pull request.

## Predictor models

CayleyPy contains a library of machine learning models to be used as predictors in the beam search algorithm for
finding paths in Cayley graph. These models can be easily accessed using `Predictor.pretrained`
([example](https://www.kaggle.com/code/fedimser/lrx-solution-n-32-beamsearch)).

Each such model is a PyTorch neural network which consists of 3 parts: 
* Model architecture description (a subclass of `nn.Models`) - defined in `cayleypy/models.py`.
* Model architecture hyperparameters (such as input size or sizes of hidden layers) - defined by `models.ModelConfig`.
* Model weights - these are stored on Kaggle.

List of currently available models is 
[here](https://github.com/cayleypy/cayleypy/blob/main/cayleypy/models/models_lib.py).

### How to add a new predictor model
1. Train your model.
2. Verify that when used with beam search, it reliably finds the paths.
3. Export weights to a file (using `torch.save(model.state_dict(), path)`.
4. Upload weights as model on Kaggle, make it public and use opensource license (MIT license is recommended).
5. Make sure the graph for which your model should be used has unique name (that is, `CayleyGraphDef.name`). For
    example, `PermutationGroups.lrx(16)` has name "lrx-16". Also `prepare_graph` given this name should return
    this graph (this is needed for tests).
6. Define `ModelConfig` for your model:
    * `weights_kaggle_id` is identifier of your saved model on Kaggle. This is what you would pass to 
      `kagglehub.model_download`.
    * `weights_path` is the name of file with weights.
    * If your can be exactly described by one of available model types in `models/models.py`, use that model type
        with appropriate hyperparameters. If needed, add new hyperparameters to ModelConfig.
    * If your model architecture is very different from we already have in library, define new model type for it.
    * For example, we already have model type "MLP" (multi-layer perceptron) defined by `MlpModel` with the following
        parameters: `input_size`, `num_classes_for_one_hot`, `layers_sizes`.
7. Verify that when you define your model config, call `load` on it and then use that as preditor in beam search,
    it works.
8. Add your model to `PREDICTOR_MODELS` in `models_lib`. Use graph name as a key.
9. Run `pytest cayleypy/models/models_lib_test.py`. This will check that your model can be loaded from Kaggle and used
    for inference (i.e. has correct input and output shape), but it doesn't check quality of your model.
9. Optionally, add a test that beam search with your model successfully finds a path.

## Credits

The idea of the project - Alexander Chervov - see https://arxiv.org/abs/2502.18663, 
https://arxiv.org/abs/2502.13266, discussion group https://t.me/sberlogasci/1,
Early ideas and prototypes appeared during Kaggle challenge Santa 2023:
Prototype: https://www.kaggle.com/code/alexandervc/santa23-globe26-modeling5,
Description: https://www.kaggle.com/competitions/santa-2023/discussion/466399, 
https://www.kaggle.com/competitions/santa-2023/discussion/472594. 

The initial code developments can be found at Kaggle dataset:
https://www.kaggle.com/datasets/alexandervc/growth-in-finite-groups (see paper https://arxiv.org/abs/2502.13266 )
Other developments can be found at:
https://www.kaggle.com/competitions/lrx-oeis-a-186783-brainstorm-math-conjecture/code,
https://www.kaggle.com/datasets/alexandervc/cayleypy-development-3-growth-computations,
see also beam-search part: [ Cayleypy (Ivan Koltsov) ](https://github.com/iKolt/cayleypy),
Rubik's cube part: [Piligrim (Kirill Khoruzhii)](https://github.com/k1242).

Also, code from the following Kaggle notebooks was used:

* https://www.kaggle.com/code/ivankolt/generation-of-incidence-mtx-pancake (advanced BFS).
* https://www.kaggle.com/code/avm888/cayleypy-growth-function.
* https://www.kaggle.com/code/avm888/jax-version-cayleypy (how to use JAX).
* https://www.kaggle.com/code/fedimser/bfs-for-binary-string-permutations (bit operations).
* https://www.kaggle.com/code/ivankolt/lrx-4bit-uint64?scriptVersionId=221435319 (fast BFS)
