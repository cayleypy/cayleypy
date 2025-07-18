from cayleypy.puzzles.gap_to_graph import (
    gap_to_CayleyGraphDef,
    get_gaps_dir,
    cayley_graph_for_puzzle_gap,
    list_gap_puzzles_defaults,
)


def test_gap_dirs():
    gaps_dir = get_gaps_dir()
    assert gaps_dir.exists(), f"Gaps directory: {gaps_dir} does not exist."
    assert (gaps_dir / "defaults").exists(), f"Gaps defaults directory: {gaps_dir / 'defaults'} does not exist."


def test_default_gaps():
    puzzles = list_gap_puzzles_defaults()
    assert len(puzzles) > 0, "No puzzles found in the default GAP puzzles directory."
    for puzzle in puzzles:
        cayley_graph_def = cayley_graph_for_puzzle_gap(puzzle)
        assert len(cayley_graph_def.generators) > 0, f"Cayley graph for {puzzle} doesn't have any generators."
