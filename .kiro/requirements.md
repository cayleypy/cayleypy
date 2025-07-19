# Requirements Document

## Introduction

CayleyPy is a Python library for analyzing extremely large state-space graphs, particularly Cayley graphs and Schreier coset graphs. The library represents combinatorial puzzles and mathematical structures as graphs where vertices are group elements (states) and edges represent group generators (valid moves). This mathematical foundation enables efficient analysis of puzzles with astronomical state spaces that cannot be stored in memory, such as Rubik's Cube with approximately 4.3×10^19 states.

## Requirements

### Requirement 1: Graph Definition and State Space Representation

**User Story:** As a researcher, I want to define puzzles and combinatorial structures as Cayley graphs using mathematical group theory, so that I can analyze their properties using rigorous mathematical foundations.

#### Acceptance Criteria

1. WHEN a puzzle is defined using permutation generators, THE SYSTEM SHALL create a CayleyGraphDef with GeneratorType.PERMUTATION that represents states as permutations and moves as permutation operations
2. WHEN a puzzle is defined using matrix generators, THE SYSTEM SHALL create a CayleyGraphDef with GeneratorType.MATRIX that represents states as matrices and moves as matrix multiplication operations
3. WHEN generators are provided for a puzzle, THE SYSTEM SHALL validate that permutation generators are valid permutations and matrix generators are square matrices of consistent dimensions
4. WHEN a central state is specified, THE SYSTEM SHALL use it as the reference point for all graph computations, defaulting to the identity element if not provided
5. WHEN generators form an inverse-closed set, THE SYSTEM SHALL automatically detect this property and optimize algorithms accordingly

### Requirement 2: Efficient State Space Exploration

**User Story:** As a puzzle solver, I want to explore large state spaces efficiently without running out of memory, so that I can analyze puzzles with millions or billions of states.

#### Acceptance Criteria

1. WHEN performing breadth-first search on a graph, THE SYSTEM SHALL explore states layer by layer while maintaining configurable memory limits to prevent system crashes
2. WHEN the state space is too large to store completely, THE SYSTEM SHALL provide options to store only specific layers (first, last, or layers below a size threshold)
3. WHEN exploring states, THE SYSTEM SHALL use efficient hash-based deduplication to avoid revisiting previously seen states
4. WHEN memory usage approaches configured limits, THE SYSTEM SHALL automatically trigger garbage collection and memory cleanup procedures
5. WHEN generators are inverse-closed, THE SYSTEM SHALL optimize memory usage by storing only the last two layers of visited states during BFS

### Requirement 3: Hardware-Accelerated Computation

**User Story:** As a performance-conscious researcher, I want to leverage GPU/TPU hardware acceleration for large-scale computations, so that I can analyze complex puzzles faster than CPU-only implementations.

#### Acceptance Criteria

1. WHEN GPU hardware is available and device is set to "auto" or "cuda", THE SYSTEM SHALL automatically utilize CUDA acceleration for tensor operations
2. WHEN performing batch operations on states, THE SYSTEM SHALL use vectorized PyTorch operations to process multiple states simultaneously
3. WHEN memory limits are configured, THE SYSTEM SHALL monitor GPU memory usage and perform device-specific cleanup when thresholds are exceeded
4. WHEN bit encoding is enabled, THE SYSTEM SHALL compress state representations to optimize memory usage and transfer speeds
5. WHEN TPU acceleration is requested, THE SYSTEM SHALL provide JAX-compatible interfaces for XLA compilation and TPU execution

### Requirement 4: Advanced Search Algorithms

**User Story:** As an algorithm researcher, I want access to multiple search strategies including heuristic-guided search, so that I can find solutions efficiently and study different exploration patterns.

#### Acceptance Criteria

1. WHEN no heuristic is provided, THE SYSTEM SHALL use breadth-first search to guarantee finding shortest paths in unweighted graphs
2. WHEN a heuristic predictor is provided, THE SYSTEM SHALL use beam search with configurable beam width to focus exploration on promising states
3. WHEN performing beam search, THE SYSTEM SHALL support custom predictors including neural networks, built-in heuristics (Hamming distance), and user-defined functions
4. WHEN path reconstruction is requested, THE SYSTEM SHALL maintain predecessor information and provide methods to reconstruct solution sequences
5. WHEN random sampling is needed, THE SYSTEM SHALL provide both classic random walks and BFS-guided sampling with configurable parameters

### Requirement 5: Puzzle Integration and Extensibility

**User Story:** As a puzzle enthusiast, I want to easily work with predefined puzzles and add new puzzle types, so that I can solve various combinatorial problems without implementing low-level graph operations.

#### Acceptance Criteria

1. WHEN working with permutation-based puzzles, THE SYSTEM SHALL provide factory methods in PermutationGroups class for mathematical permutation groups
2. WHEN working with matrix-based puzzles, THE SYSTEM SHALL provide factory methods in MatrixGroups class for linear algebra structures
3. WHEN working with physical puzzles, THE SYSTEM SHALL provide factory methods in Puzzles class for real-world puzzles like Rubik's Cube, Globe puzzle, and Hungarian Rings
4. WHEN adding new puzzles, THE SYSTEM SHALL allow extension through creating new CayleyGraphDef instances following established patterns in graphs_lib.py and puzzles/puzzles.py
5. WHEN puzzle moves are complex, THE SYSTEM SHALL support storing hardcoded permutations in puzzles/moves.py and referencing them in puzzle definitions

### Requirement 6: Data Management and Analysis

**User Story:** As a data analyst, I want to store, load, and analyze computational results from large graph explorations, so that I can build upon previous work and share findings with others.

#### Acceptance Criteria

1. WHEN BFS exploration is complete, THE SYSTEM SHALL provide growth functions showing the number of states at each distance from the central state
2. WHEN results need to be saved, THE SYSTEM SHALL support exporting to standard formats including CSV for growth data and JSON for configuration data
3. WHEN loading precomputed datasets, THE SYSTEM SHALL validate data integrity and provide access through the datasets module
4. WHEN working with large results, THE SYSTEM SHALL support result caching to avoid recomputing expensive operations
5. WHEN integration with external tools is needed, THE SYSTEM SHALL provide NetworkX export capabilities for smaller graphs

### Requirement 7: Performance Monitoring and Optimization

**User Story:** As a performance engineer, I want detailed control over computational parameters and monitoring capabilities, so that I can optimize performance for specific hardware and problem sizes.

#### Acceptance Criteria

1. WHEN configuring computation parameters, THE SYSTEM SHALL provide options for batch size, memory limits, hash chunk size, and bit encoding width
2. WHEN verbose logging is enabled, THE SYSTEM SHALL provide detailed progress information including layer sizes, memory usage, and timing information
3. WHEN processing large batches, THE SYSTEM SHALL automatically split work into manageable chunks based on configured batch sizes
4. WHEN using string encoding for permutations, THE SYSTEM SHALL automatically determine optimal bit widths or allow manual specification
5. WHEN random operations are performed, THE SYSTEM SHALL support deterministic behavior through configurable random seeds

### Requirement 8: Mathematical Correctness and Validation

**User Story:** As a mathematician, I want assurance that all computations are mathematically correct and that the library handles edge cases properly, so that I can trust the results for research and publication.

#### Acceptance Criteria

1. WHEN generators are not inverse-closed, THE SYSTEM SHALL correctly handle asymmetric graphs and provide appropriate warnings or limitations
2. WHEN matrix operations involve modular arithmetic, THE SYSTEM SHALL correctly implement modular matrix multiplication with specified moduli
3. WHEN path reconstruction is requested, THE SYSTEM SHALL validate that reconstructed paths actually transform the start state to the goal state
4. WHEN working with different generator types, THE SYSTEM SHALL prevent mixing permutation and matrix generators in the same graph definition
5. WHEN mathematical properties are queried, THE SYSTEM SHALL provide accurate information about inverse-closure, generator counts, and state space dimensions

### Requirement 9: Error Handling and Robustness

**User Story:** As a library user, I want clear error messages and graceful handling of invalid inputs, so that I can quickly identify and fix problems in my code.

#### Acceptance Criteria

1. WHEN invalid generators are provided, THE SYSTEM SHALL raise clear exceptions explaining what makes the generators invalid
2. WHEN memory limits are exceeded despite configuration, THE SYSTEM SHALL provide informative warnings and attempt graceful degradation
3. WHEN hardware acceleration fails, THE SYSTEM SHALL automatically fall back to CPU computation with appropriate logging
4. WHEN file operations fail, THE SYSTEM SHALL provide specific error messages indicating the nature of the failure
5. WHEN mathematical operations are invalid (e.g., non-invertible matrices), THE SYSTEM SHALL detect and report these conditions clearly

### Requirement 10: Code Quality and Development Standards

**User Story:** As a developer contributing to CayleyPy, I want mandatory code quality checks to ensure consistent, maintainable, and reliable code, so that the library maintains high standards and prevents regressions.

#### Acceptance Criteria

1. WHEN any code changes are made, THE SYSTEM SHALL require Black code formatting with 120-character line limits and all formatting checks must pass before commits
2. WHEN code is committed, THE SYSTEM SHALL require pylint score of 10.00/10 and all mypy type checking errors must be resolved
3. WHEN new functionality is added, THE SYSTEM SHALL require comprehensive unit tests with pytest that achieve full code coverage for the new features
4. WHEN tests are run, THE SYSTEM SHALL execute `RUN_SLOW_TESTS=1 pytest` and all tests must pass without failures, timeouts, or memory issues
5. WHEN matrix group functionality is tested, THE SYSTEM SHALL use finite moduli (3, 5, 7) instead of infinite groups (modulo=0) and limit BFS diameters to 1-3 to prevent performance issues
6. WHEN performance-critical code is modified, THE SYSTEM SHALL include benchmark tests using pytest-benchmark to detect performance regressions
7. WHEN commits are made, THE SYSTEM SHALL require the pre-commit workflow: `black .` → `./lint.sh` → `RUN_SLOW_TESTS=1 pytest` with all checks passing
8. WHEN pylint warnings occur for intentional patterns (like import-outside-toplevel), THE SYSTEM SHALL include appropriate disable comments following existing codebase patterns