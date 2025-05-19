from cayleypy import prepare_graph


def test_all_transpositions():
    graph = prepare_graph("all_transpositions", n=20)
    assert graph.n_generators == (20 * 19) // 2


def test_cube333():
    graph = prepare_graph("cube_3/3/3_12gensQTM")
    assert graph.n_generators == 12

    graph = prepare_graph("cube_3/3/3_18gensHTM")
    assert graph.n_generators == 18
