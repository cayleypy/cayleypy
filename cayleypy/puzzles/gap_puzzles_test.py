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


def test_inverse_closed():
    graph1 = GapPuzzles.puzzle("2x2x2")
    assert graph1.name == "2x2x2.gap-ic"
    assert graph1.n_generators == 12
    graph2 = GapPuzzles.puzzle("2x2x2", make_inverse_closed=False)
    assert graph2.name == "2x2x2.gap"
    assert graph2.n_generators == 6
