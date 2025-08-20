from cayleypy import CayleyGraph, PermutationGroups, load_dataset, MatrixGroups
from cayleypy.algo import InteractiveBfs


def _growth_using_interactive_bfs(graph):
    ibfs = InteractiveBfs(graph, graph.central_state)
    ans = []
    while len(ibfs.cur_layer) > 0:
        ans.append(len(ibfs.cur_layer))
        ibfs.step()
    return ans


def test_interactive_bfs_lx_8():
    g = CayleyGraph(PermutationGroups.lx(8))
    assert _growth_using_interactive_bfs(g) == load_dataset("lx_cayley_growth")["8"]


def test_interactive_bfs_lrx_coset_16():
    start_state = "01" * 8
    g = CayleyGraph(PermutationGroups.lrx(16).with_central_state(start_state))
    assert _growth_using_interactive_bfs(g) == load_dataset("lrx_coset_growth")[start_state]


def test_interactive_bfs_coxeter_8():
    g = CayleyGraph(PermutationGroups.coxeter(8))
    assert _growth_using_interactive_bfs(g) == load_dataset("coxeter_cayley_growth")["8"]


def test_interactive_bfs_heisenberg():
    g = CayleyGraph(MatrixGroups.heisenberg(modulo=10))
    assert _growth_using_interactive_bfs(g) == load_dataset("heisenberg_growth")["3,10"]
