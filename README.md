# CaleyPy

Library for studying Cayley graphs and Schreier coset graphs.

## Overview

This is a library for studying
[Schreier coset graphs](https://en.wikipedia.org/wiki/Schreier_coset_graph)
and [Caley graphs](https://en.wikipedia.org/wiki/Cayley_graph).

This what we plan to support:

* BFS from given state (growth function, adjacency matrix, last layers).
* Efficient path finding (beam search).
* Random walk generation.
* Library of graphs and generators (LRX, TopSpin, Rubik Cubes, wreath, globe etc.,
  see [here](https://www.kaggle.com/code/ivankolt/generation-of-incidence-mtx-pancake)).
* Efficient computing on CPU, GPU, TPU (with JAX), usable on Kaggle.
* Library of datasets with solutions to some problems (e.g. growth functions like
  [here](https://www.kaggle.com/code/fedimser/bfs-for-binary-string-permutations)).

## Usage

See this demo [Kaggle notebook](https://www.kaggle.com/code/fedimser/cayleypy-demo) for examples
on how this library can be used.

## Development

To start development, run:

```
git clone https://github.com/cayleypy/cayleypy.git
cd cayleypy
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

To run only quick tests:

```
FAST=1 pytest
```

Before commiting, run these checks:

```
tools/lint.sh
pytest 
```

To check coverage, run:

```
coverage run -m pytest && coverage html
```

## Credits

The initial code for this library is based on [cayleypy](https://github.com/iKolt/cayleypy)
by [Ivan Koltsov](https://github.com/iKolt), which is itself based on code by
[Alexander Chervov](https://github.com/chervov) and
[Kirill Khoruzhii](https://github.com/k1242).

Also, code from the following Kaggle notebooks was used:

* https://www.kaggle.com/code/ivankolt/generation-of-incidence-mtx-pancake (advanced BFS).
* https://www.kaggle.com/code/avm888/cayleypy-growth-function.
* https://www.kaggle.com/code/avm888/jax-version-cayleypy (how to use JAX).
* https://www.kaggle.com/code/fedimser/bfs-for-binary-string-permutations (bit operations).
* https://www.kaggle.com/code/ivankolt/lrx-4bit-uint64?scriptVersionId=221435319 (fast BFS)
