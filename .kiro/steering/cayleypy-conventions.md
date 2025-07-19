---
inclusion: always
---

# CayleyPy Development Guidelines

CayleyPy is a Python library for analyzing extremely large graphs using mathematical group theory and AI/ML methods. Focus on state-space graphs too large for memory storage.

## Code Standards

- Black formatter with 120-character lines
- Type hints required (mypy configured)
- Docstrings mandatory for public APIs
- Descriptive variable names for mathematical concepts
- PEP 8 compliance

## Mandatory Pre-Commit Quality Checks

**CRITICAL: Before every commit, you MUST run and fix all issues from:**

1. **Code Formatting:** `black --check --diff .`
   - Fix any formatting issues with: `black .`
   - All files must pass Black formatting standards

2. **Linting:** `./lint.sh`
   - Must achieve 10.00/10 pylint score
   - Fix all mypy type checking errors
   - Address all code quality warnings

3. **Testing:** `RUN_SLOW_TESTS=1 pytest`
   - All tests must pass (no failures)
   - New functionality requires comprehensive unit tests
   - Performance-critical code needs benchmark tests
   - Matrix groups: use finite moduli (avoid modulo=0)
   - Keep test diameters small (1-3) for large state spaces

**Workflow:**
```bash
# 1. Format code
black .

# 2. Run linting
./lint.sh

# 3. Run comprehensive tests
RUN_SLOW_TESTS=1 pytest

# 4. Only commit if all checks pass
git add .
git commit -m "Your commit message"
```

**Performance Guidelines for Tests:**
- Matrix group tests: use modulo 3, 5, 7 (not 0) to keep state spaces finite
- BFS tests: max_diameter should be 1-3 for matrix groups
- Avoid infinite groups in automated tests
- Test timeouts indicate performance issues that must be fixed

## Architecture Rules

### Graph Definitions

- Return `CayleyGraphDef` objects for new Cayley graphs
- Permutation graphs → `PermutationGroups` class in `graphs_lib.py`
- Matrix graphs → `MatrixGroups` class in `graphs_lib.py`
- Puzzle graphs → `Puzzles` class in `puzzles/puzzles.py`
- Complex puzzle logic → separate files in `puzzles/`
- Hardcoded moves → `puzzles/moves.py`

### Design Patterns

- Abstract base classes for core interfaces (e.g., `Puzzle`)
- Hashable state representations (tuples, strings)
- Separate puzzle logic from solving algorithms
- Support BFS and A\* search strategies

### Performance

- NumPy arrays for numerical work
- Numba for performance-critical loops
- JAX/TPU support for large-scale computation
- Efficient bit manipulation for states
- Vectorized operations preferred

## Testing

- pytest with unit tests for all new functionality
- pytest-benchmark for performance testing
- Test edge cases and boundaries
- Use `RUN_SLOW_TESTS=1` for comprehensive tests
- Maintain coverage with `coverage` tool

## Mathematical Conventions

- Consistent state representation (e.g., 0 for blank tiles)
- Admissible heuristics for optimal A\* search
- Descriptive generator names following domain conventions
- Mathematical references in docstrings (Wikipedia, arXiv)

## File Organization

- Data files in `cayleypy/data/` with descriptive CSV names
- Test files with `_test.py` suffix
- Clean separation between core library and examples

## Dependencies

- Python 3.9+ support required
- Core: numpy, scipy, numba, h5py
- Optional dependencies grouped by functionality
- Graceful handling of missing optional dependencies

## Error Handling

- Clear error messages for invalid definitions
- Graceful TPU/JAX dependency handling
- Input validation for parameters and states
- Appropriate exception types for different conditions
