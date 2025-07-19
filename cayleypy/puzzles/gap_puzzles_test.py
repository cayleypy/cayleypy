from cayleypy.puzzles import GapPuzzles


def test_list_puzzles():
    puzzle_names = GapPuzzles.list_puzzles()
    assert "2x2x2" in puzzle_names
    assert "dino" in puzzle_names


def test_load_all_puzzles():
    puzzle_names = GapPuzzles.list_puzzles()
    for puzzle_name in puzzle_names:
        graph = GapPuzzles.puzzle(puzzle_name)
        assert len(graph.generators) > 0
