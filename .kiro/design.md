# CayleyPy Design Document

## Overview

CayleyPy is designed as a high-performance library for analyzing extremely large state-space graphs, particularly Cayley graphs and Schreier coset graphs. The architecture leverages mathematical group theory to represent puzzles and combinatorial problems as graphs where vertices are states and edges represent valid moves (group generators). The system is built around three core components: graph definition (`CayleyGraphDef`), graph computation (`CayleyGraph`), and specialized algorithms for search and analysis.

## Graph-Theoretic Foundation

The core abstraction in CayleyPy is the **Cayley graph**, where:

* **States as Group Elements:** Each puzzle state is represented as a group element (permutation or matrix), enabling mathematical operations and efficient state manipulation.
* **Moves as Generators:** Valid puzzle moves are represented as group generators that transform one state to another through group operations.
* **Central State:** Every graph has a designated "central state" (typically the solved state) from which all reachable states are computed.
* **Graph Structure:** The complete state space forms a graph where vertices are states and edges represent generator applications.

This mathematical foundation allows CayleyPy to handle puzzles with astronomical state spaces (like Rubik's Cube with ~4.3×10^19 states) that cannot be stored in memory.

## Architecture Components

### CayleyGraphDef: Graph Definition Layer

The `CayleyGraphDef` class encapsulates the mathematical definition of a puzzle:

* **Generator Types:** Supports both permutation-based puzzles (sliding puzzles, twisty puzzles) and matrix-based puzzles (linear transformations).
* **Generator Specification:** Defines the set of valid moves as mathematical generators, with support for inverse-closed generator sets.
* **State Representation:** Specifies how puzzle states are encoded (as permutations for most puzzles, or as matrices for algebraic puzzles).
* **Extensibility:** New puzzles are added by creating `CayleyGraphDef` instances with appropriate generators, following the established patterns in `graphs_lib.py` and `puzzles/puzzles.py`.

The design separates puzzle definition from computation, allowing the same solving algorithms to work across all puzzle types without modification.

### CayleyGraph: Computational Engine

The `CayleyGraph` class provides the computational interface for graph operations:

* **State Encoding:** Converts between human-readable states and internal representations optimized for computation.
* **Neighbor Generation:** Efficiently computes all states reachable from a given state by applying generators.
* **Memory Management:** Handles large-scale computations with configurable memory limits and garbage collection.
* **Hardware Acceleration:** Supports GPU/TPU acceleration through PyTorch tensors and optional JAX integration.

### Search Algorithms

CayleyPy implements several search strategies optimized for large state spaces:

#### Breadth-First Search (BFS)
* **Layer-by-layer exploration:** Visits states in order of distance from the central state.
* **Memory optimization:** Configurable layer storage limits to handle graphs too large for memory.
* **Growth analysis:** Computes growth functions showing the number of states at each distance.
* **Complete graph construction:** Can build explicit representations of smaller graphs.

#### Beam Search
* **Heuristic-guided exploration:** Uses predictors to focus search on promising states.
* **Configurable beam width:** Balances exploration breadth with computational efficiency.
* **Path reconstruction:** Maintains predecessor information for solution recovery.
* **Integration with ML models:** Supports neural network heuristics through the `Predictor` class.

#### Random Walks
* **Statistical sampling:** Generates representative samples from large state spaces.
* **Multiple modes:** Supports both classic random walks and BFS-guided sampling.
* **Distance estimation:** Provides approximate distance information for sampled states.

## Puzzle Integration Architecture

### Puzzle Definition Patterns

CayleyPy organizes puzzle definitions into logical categories:

* **PermutationGroups:** Mathematical permutation groups (in `graphs_lib.py`)
* **MatrixGroups:** Linear algebra-based puzzles (in `graphs_lib.py`)
* **Physical Puzzles:** Real-world puzzles like Rubik's Cube, Globe puzzle (in `puzzles/puzzles.py`)

### Puzzle Implementation Structure

Each puzzle follows a consistent pattern:

1. **Generator Definition:** Specify the mathematical generators (permutations or matrices)
2. **Central State:** Define the solved/reference state
3. **Naming Convention:** Provide descriptive names for generators
4. **Factory Method:** Create static methods in appropriate classes for instantiation

### Extensibility Framework

Adding new puzzles involves:

* **Simple Puzzles:** Add generator definitions to existing classes
* **Complex Puzzles:** Create dedicated modules in `puzzles/` directory
* **Move Definitions:** Store hardcoded permutations in `puzzles/moves.py`
* **Testing:** Include comprehensive unit tests with performance benchmarks

## Performance Optimization Design

### Memory Efficiency

* **Bit Encoding:** Compress state representations using configurable bit widths
* **String Encoding:** Efficient encoding for permutation-based states
* **Chunked Processing:** Process large datasets in configurable chunks
* **Garbage Collection:** Automatic memory management with device-specific optimizations

### Hardware Acceleration

* **PyTorch Integration:** Leverage GPU acceleration for tensor operations
* **Device Abstraction:** Automatic device selection (CPU/CUDA) with manual override
* **Batch Processing:** Vectorized operations for neighbor generation and state manipulation
* **Memory Monitoring:** Configurable memory limits with automatic cleanup

### Algorithmic Optimizations

* **Hash-based Deduplication:** Efficient duplicate removal using custom hashing
* **Inverse-closed Optimization:** Special handling for symmetric generator sets
* **Batched Neighbor Generation:** Parallel computation of state transitions
* **Configurable Precision:** Trade-off between accuracy and performance

## Data Management and I/O

### Dataset Integration

* **Precomputed Results:** Store and load growth functions and analysis results
* **CSV Format:** Standardized format for growth data and experimental results
* **Compression:** Efficient storage of large datasets
* **Validation:** Automatic verification of loaded data integrity

### File Organization

* **Data Directory:** Centralized storage in `cayley/data/` with descriptive naming
* **Result Caching:** Automatic caching of expensive computations
* **Export Capabilities:** Support for various output formats (CSV, JSON, NetworkX)

## Testing and Validation Framework

### Test Categories

* **Unit Tests:** Individual component testing with `pytest`
* **Performance Tests:** Benchmarking with `pytest-benchmark`
* **Integration Tests:** End-to-end puzzle solving validation
* **Slow Tests:** Comprehensive testing with `RUN_SLOW_TESTS=1`

### Quality Assurance

* **Code Coverage:** Maintain high test coverage with the `coverage` tool
* **Type Checking:** Static analysis with `mypy`
* **Code Formatting:** Consistent style with Black formatter (120 char lines)
* **Documentation:** Comprehensive docstrings with mathematical references

## Future Extensibility

### Machine Learning Integration

* **Predictor Framework:** Pluggable heuristic functions for guided search
* **Neural Network Support:** Integration with PyTorch models for state evaluation
* **Training Pipeline:** Support for learning heuristics from puzzle data

### Advanced Algorithms

* **Pattern Databases:** Precomputed heuristic tables for complex puzzles
* **Bidirectional Search:** Search from both start and goal states
* **Parallel Search:** Multi-threaded exploration of state spaces
* **Distributed Computing:** Support for cluster-based computations

### Hardware Scaling

* **TPU Support:** Tensor Processing Unit acceleration for massive computations
* **JAX Integration:** XLA compilation for optimized execution
* **Multi-GPU:** Distributed computation across multiple devices
* **Cloud Integration:** Support for cloud-based high-performance computing

This design ensures CayleyPy can scale from simple educational puzzles to research-grade analysis of complex combinatorial structures while maintaining mathematical rigor and computational efficiency.