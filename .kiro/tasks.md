# Implementation Plan

- [ ] 1. Create Puzzle abstraction layer for CayleyPy integration
  - Implement a `Puzzle` base class that wraps `CayleyGraphDef` and `CayleyGraph` functionality
  - Define standard interface methods: `get_neighbors(state)`, `is_goal(state)`, `heuristic(state)`
  - Provide state conversion utilities between puzzle-specific and CayleyPy internal representations
  - _Requirements: WHEN given an initial puzzle state and a goal state, THE SYSTEM SHALL compute a sequence of valid moves_

- [ ] 2. Implement EightPuzzle as reference implementation
  - Create `EightPuzzle` class extending the `Puzzle` base class
  - Define 8-puzzle state representation (3x3 grid with 0 as blank tile)
  - Implement Manhattan distance heuristic for A* search optimization
  - Generate appropriate permutation generators for sliding tile moves
  - _Requirements: WHEN a heuristic function is provided, THE SYSTEM SHALL perform heuristic-guided search_

- [ ] 3. Develop unified solver interface
  - Create `PuzzleSolver` class that bridges puzzle abstraction with CayleyPy algorithms
  - Implement BFS solving using CayleyPy's existing BFS functionality
  - Add A* search capability using CayleyPy's beam search with heuristic integration
  - Ensure solution path reconstruction returns move sequences
  - _Requirements: WHEN no heuristic is provided, THE SYSTEM SHALL use uninformed search strategy_

- [ ] 4. Implement solution path reconstruction
  - Extend CayleyPy's path finding capabilities for puzzle-specific move sequences
  - Convert internal generator sequences to human-readable move descriptions
  - Handle both forward and backward path reconstruction
  - Validate solution paths by applying moves to initial states
  - _Requirements: WHEN a solution is found, THE SYSTEM SHALL return the solution as an ordered list of moves_

- [ ] 5. Add puzzle definition file support
  - Design JSON schema for puzzle configuration files
  - Implement `PuzzleLoader` class for parsing puzzle definition files
  - Create factory pattern for instantiating puzzle types from file specifications
  - Support parameterized puzzle creation (e.g., different cube sizes, grid dimensions)
  - _Requirements: WHEN a puzzle definition is provided via external file, THE SYSTEM SHALL parse and load the puzzle_

- [ ] 6. Integrate TPU acceleration support
  - Extend existing CayleyPy TPU capabilities for puzzle solving
  - Implement JAX-compatible state representations for supported puzzle types
  - Create vectorized neighbor generation functions for batch processing
  - Add TPU-optimized solver mode with fallback to CPU implementation
  - _Requirements: WHEN configured to use TPU acceleration, THE SYSTEM SHALL offload computations to TPU_

- [ ] 7. Implement comprehensive testing suite
  - Create unit tests for Puzzle base class and EightPuzzle implementation
  - Test solver correctness with known puzzle solutions
  - Validate heuristic effectiveness by comparing BFS vs A* performance
  - Test file loading with valid and invalid puzzle definitions
  - Add performance benchmarks for TPU vs CPU solving
  - _Requirements: Ensure system avoids revisiting states and handles no-solution cases_

- [x] 8. Fix BfsResult persistence for matrix groups ([GitHub Issue #59](https://github.com/cayleypy/cayleypy/issues/59))
  - Extend `BfsResult.save()` to properly serialize matrix generators and their moduli to HDF5 format
  - Update `BfsResult.load()` to detect matrix-type graphs and use `CayleyGraphDef.for_matrix_group()` constructor
  - Ensure matrix generator data (matrices and modulus values) are correctly saved as separate HDF5 datasets
  - Fix graph reconstruction to create `MatrixGenerator` objects from loaded matrix data
  - Validate `BfsResult.__eq__()` works correctly for matrix groups after save/load round-trip
  - Add comprehensive unit tests for matrix group BFS save/load using `MatrixGroups.heisenberg()` or similar
  - Test edge cases: matrix groups with different moduli, various matrix sizes, and BFS with all optional data
  - _Requirements: WHEN BFS results from matrix groups are saved and loaded, THE SYSTEM SHALL preserve all data and maintain equality_

- [ ] 9. Create documentation and usage examples
  - Write comprehensive API documentation for puzzle solving interface
  - Create tutorial showing how to define and solve custom puzzles
  - Document TPU setup and configuration requirements
  - Provide example puzzle definition files and solving scripts
  - Include performance optimization guidelines
  - _Requirements: Support extensibility for new puzzle types through well-defined interface_