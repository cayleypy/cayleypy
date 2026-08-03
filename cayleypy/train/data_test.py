import pytest
import torch

from .data import (
    BfsAnchors,
    DataSource,
    MixtureDataSource,
    PathDataSource,
    RandomWalksSource,
    SparseQSampler,
    TrainingData,
)
from ..cayley_graph import CayleyGraph
from ..cayley_path import CayleyPath
from ..graphs_lib import PermutationGroups

# Number of states in the whole graph returned by _lrx5 (5!).
LRX5_SIZE = 120


def _lrx5() -> CayleyGraph:
    return CayleyGraph(PermutationGroups.lrx(5), device="cpu")


def _exact_distances(graph: CayleyGraph) -> dict[tuple, int]:
    """Returns true distance of every state of the graph (as a map from the state to it), computed by exact BFS."""
    layers = graph.bfs(max_layer_size_to_store=None).layers
    return {tuple(state.tolist()): distance for distance, layer in layers.items() for state in layer}


class _ConstantSource(DataSource):
    """Data source generating a fixed number of states, whose targets say which source they came from."""

    def __init__(self, n_states: int, tag: float, state_size: int = 5):
        self.n_states = n_states
        self.tag = tag
        self.state_size = state_size

    def generate(self) -> TrainingData:
        states = torch.full((self.n_states, self.state_size), int(self.tag), dtype=torch.int64)
        return TrainingData(states=states, targets=torch.full((self.n_states,), self.tag))


def test_training_data_select():
    data = TrainingData(
        states=torch.tensor([[0, 1], [1, 0], [0, 0]]),
        targets=torch.tensor([0.0, 1.0, 2.0]),
        mask=torch.tensor([True, False, True]),
        weights=torch.tensor([1.0, 0.5, 0.25]),
    )
    assert len(data) == 3
    assert data.n_outputs == 1

    selected = data.select(torch.tensor([2, 0]))
    assert torch.equal(selected.states, torch.tensor([[0, 0], [0, 1]]))
    assert torch.equal(selected.targets, torch.tensor([2.0, 0.0]))
    assert torch.equal(selected.mask, torch.tensor([True, True]))
    assert torch.equal(selected.weights, torch.tensor([0.25, 1.0]))


def test_training_data_reports_number_of_outputs():
    states = torch.zeros((4, 5), dtype=torch.int64)
    assert TrainingData(states, torch.zeros(4)).n_outputs == 1
    assert TrainingData(states, torch.zeros((4, 3))).n_outputs == 3


def test_training_data_concat_fills_missing_mask_and_weights():
    states = torch.zeros((2, 5), dtype=torch.int64)
    sparse = TrainingData(states, torch.ones(2), mask=torch.tensor([True, False]), weights=torch.tensor([0.5, 0.5]))
    dense = TrainingData(states, torch.zeros(2))
    data = TrainingData.concat([sparse, dense])

    assert len(data) == 4
    assert torch.equal(data.targets, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    # Data that has no mask (or weights) is fully labeled with weight 1, and stays that way when it is mixed in.
    assert torch.equal(data.mask, torch.tensor([True, False, True, True]))
    assert torch.equal(data.weights, torch.tensor([0.5, 0.5, 1.0, 1.0]))

    without_mask = TrainingData.concat([dense, dense])
    assert without_mask.mask is None and without_mask.weights is None


def test_training_data_rejects_inconsistent_shapes():
    states = torch.zeros((3, 5), dtype=torch.int64)
    with pytest.raises(ValueError, match=r"states must have shape \[n_states, state_size\]"):
        TrainingData(torch.zeros(5, dtype=torch.int64), torch.zeros(1))
    with pytest.raises(ValueError, match="targets must have shape"):
        TrainingData(states, torch.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match="There are 3 states, but 2 targets for them"):
        TrainingData(states, torch.zeros(2))
    with pytest.raises(ValueError, match="Shape of mask is"):
        TrainingData(states, torch.zeros(3), mask=torch.ones((3, 2), dtype=torch.bool))
    with pytest.raises(ValueError, match="Shape of weights is"):
        TrainingData(states, torch.zeros(3), weights=torch.ones(2))


def test_training_data_concat_rejects_incompatible_pieces():
    states = torch.zeros((2, 5), dtype=torch.int64)
    with pytest.raises(ValueError, match="Cannot concatenate an empty list"):
        TrainingData.concat([])
    with pytest.raises(ValueError, match="targets for 1 outputs and data with targets for 3 outputs"):
        TrainingData.concat([TrainingData(states, torch.zeros(2)), TrainingData(states, torch.zeros((2, 3)))])


def test_random_walks_source():
    graph = _lrx5()
    data = RandomWalksSource(graph, n_walks=8, rw_length=4, mode="classic").generate()
    assert data.states.shape == (32, 5)
    assert data.targets.shape == (32,)
    assert data.targets.dtype == torch.float32
    assert data.mask is None and data.weights is None
    # Walks start at the central state, which is at distance 0.
    assert torch.equal(data.states[:8], graph.central_state.expand(8, 5))
    assert torch.equal(data.targets[:8], torch.zeros(8))


def test_random_walks_source_rejects_invalid_values():
    graph = _lrx5()
    with pytest.raises(ValueError, match="n_walks must be positive"):
        RandomWalksSource(graph, n_walks=0)
    with pytest.raises(ValueError, match="rw_length must be positive"):
        RandomWalksSource(graph, rw_length=0)


def test_sparse_q_sampler_labels_exactly_the_moves_along_the_walk():
    torch.manual_seed(42)
    graph = _lrx5()
    width, length, n_generators = 4, 6, 3
    data = SparseQSampler(graph, n_walks=width, rw_length=length).generate()

    assert data.states.shape == (width * length, 5)
    assert data.targets.shape == (width * length, n_generators)
    assert data.n_outputs == n_generators
    walks = data.states.reshape((length, width, 5))
    n_labeled = 0
    for step in range(length):
        for walk in range(width):
            expected = {}
            state = walks[step, walk]
            for generator in range(n_generators):
                child = graph.apply_path(state, [generator]).reshape(-1)
                if step > 0 and torch.equal(child, walks[step - 1, walk]):
                    # Going back on the walk leads to a state visited one step earlier.
                    expected[generator] = float(step - 1)
                elif step + 1 < length and torch.equal(child, walks[step + 1, walk]):
                    expected[generator] = float(step + 1)
            row = step * width + walk
            labeled = {int(i): float(data.targets[row, i]) for i in data.mask[row].nonzero().flatten()}
            assert labeled == expected, f"Wrong labels for state {state.tolist()} at step {step}."
            n_labeled += len(labeled)
    assert n_labeled > 0


def test_sparse_q_sampler_labels_two_moves_of_a_state_inside_the_walk():
    torch.manual_seed(42)
    width, length = 8, 6
    data = SparseQSampler(_lrx5(), n_walks=width, rw_length=length).generate()
    n_labeled = data.mask.sum(dim=1).reshape((length, width))

    # First and last states of a walk have only one neighbour on the walk, so only one move is labeled.
    assert torch.equal(n_labeled[0], torch.ones(width, dtype=n_labeled.dtype))
    assert torch.equal(n_labeled[-1], torch.ones(width, dtype=n_labeled.dtype))
    # Every other state has exactly 2 labeled moves - unless the walk stepped back, in which case both of them lead to
    # the same state and one label is enough.
    inside = n_labeled[1:-1]
    assert int(inside.max()) == 2
    assert float((inside == 2).to(torch.float32).mean()) > 0.5


def test_sparse_q_sampler_rejects_graph_that_is_not_inverse_closed():
    graph = CayleyGraph(PermutationGroups.lx(5), device="cpu")
    assert not graph.definition.generators_inverse_closed
    with pytest.raises(ValueError, match="not inverse closed"):
        SparseQSampler(graph)


def test_sparse_q_sampler_rejects_invalid_values():
    graph = _lrx5()
    with pytest.raises(ValueError, match="rw_length must be at least 2"):
        SparseQSampler(graph, rw_length=1)
    with pytest.raises(ValueError, match="n_walks must be positive"):
        SparseQSampler(graph, n_walks=-1)


def test_bfs_anchors_labels_are_exact_distances():
    graph = _lrx5()
    exact = _exact_distances(graph)
    data = BfsAnchors(graph, depth=3).generate()

    assert data.mask is None and data.weights is None
    assert data.n_outputs == 1
    assert len(data) == len({state for state, distance in exact.items() if distance <= 3})
    for state, target in zip(data.states, data.targets):
        assert float(target) == exact[tuple(state.tolist())]
    assert float(data.targets.max()) == 3


def test_bfs_anchors_labels_all_children_for_q_model():
    graph = _lrx5()
    exact = _exact_distances(graph)
    data = BfsAnchors(graph, depth=3, n_outputs=3).generate()

    # All children of an anchor have exact distances, so nothing is masked out.
    assert data.mask is None
    assert data.n_outputs == 3
    # Checking every column separately (rather than all of them at once) is what catches a transposed layout.
    for generator in range(3):
        children = graph.apply_path(data.states, [generator])
        expected = [exact[tuple(child.tolist())] for child in children]
        assert data.targets[:, generator].tolist() == expected


def test_bfs_anchors_for_q_model_skips_states_with_unknown_children():
    graph = _lrx5()
    exact = _exact_distances(graph)
    # The search stops at depth 3, so states at that depth have children it did not see and cannot be anchors.
    data = BfsAnchors(graph, depth=3, n_outputs=3).generate()
    assert max(exact[tuple(state.tolist())] for state in data.states) == 2

    # When the search explored the whole graph, every state can be an anchor.
    anchors = BfsAnchors(graph, depth=LRX5_SIZE, n_outputs=3)
    assert anchors.n_states_with_exact_distance == LRX5_SIZE
    assert len(anchors.generate()) == LRX5_SIZE


def test_bfs_anchors_samples_requested_number_of_states():
    graph = _lrx5()
    exact = _exact_distances(graph)
    data = BfsAnchors(graph, depth=2, size=100).generate()
    assert len(data) == 100
    assert data.targets.shape == (100,)
    for state, target in zip(data.states, data.targets):
        assert float(target) == exact[tuple(state.tolist())]


def test_bfs_anchors_stops_when_there_are_too_many_states():
    with pytest.raises(ValueError, match="more than max_table_states=5"):
        BfsAnchors(_lrx5(), depth=10, max_table_states=5)


def test_bfs_anchors_rejects_invalid_values():
    graph = _lrx5()
    with pytest.raises(ValueError, match="depth must be at least 1"):
        BfsAnchors(graph, depth=0)
    with pytest.raises(ValueError, match="size must be positive"):
        BfsAnchors(graph, depth=2, size=0)
    with pytest.raises(ValueError, match="max_table_states must be positive"):
        BfsAnchors(graph, depth=2, max_table_states=0)
    with pytest.raises(ValueError, match="n_outputs must be either 1 or the number of generators"):
        BfsAnchors(graph, depth=2, n_outputs=2)


def _solved_path(graph: CayleyGraph, start_state: list) -> CayleyPath:
    """Finds a path from `start_state` to the central state with beam search."""
    state = torch.tensor(start_state)
    result = graph.beam_search(start_state=state, return_path=True)
    assert result.path_found and result.path is not None
    return CayleyPath(state, result.path, graph.definition)


def test_path_data_source_labels_remaining_moves():
    graph = _lrx5()
    path = _solved_path(graph, [2, 0, 1, 4, 3])
    data = PathDataSource(graph, [path]).generate()

    path_length = len(path.edges)
    assert len(data) == path_length + 1
    # The state visited after p moves is (path_length - p) moves away from the central state.
    assert data.targets.tolist() == [float(path_length - p) for p in range(path_length + 1)]
    assert torch.equal(data.states[0], path.start_state)
    assert torch.equal(data.states[-1], graph.central_state)
    for p, state in enumerate(data.states):
        assert torch.equal(graph.apply_path(path.start_state, path.edges[:p]).reshape(-1), state)


def test_path_data_source_marks_targets_as_upper_bounds():
    graph = _lrx5()
    data = PathDataSource(graph, [_solved_path(graph, [2, 0, 1, 4, 3])], weight=0.25).generate()
    assert torch.equal(data.weights, torch.full_like(data.targets, 0.25))
    # By default targets weigh as much as targets of any other source.
    default_data = PathDataSource(graph, [_solved_path(graph, [2, 0, 1, 4, 3])]).generate()
    assert torch.equal(default_data.weights, torch.ones_like(default_data.targets))


def test_path_data_source_labels_the_move_of_the_path_for_q_model():
    graph = _lrx5()
    path = _solved_path(graph, [2, 0, 1, 4, 3])
    data = PathDataSource(graph, [path], n_outputs=3).generate()

    path_length = len(path.edges)
    # The central state takes no move, so it is not in the data.
    assert len(data) == path_length
    assert data.targets.shape == (path_length, 3)
    for p in range(path_length):
        labeled = data.mask[p].nonzero().flatten().tolist()
        assert labeled == [path.edges[p]]
        assert float(data.targets[p, path.edges[p]]) == path_length - p - 1


def test_path_data_source_samples_requested_number_of_states():
    graph = _lrx5()
    paths = [_solved_path(graph, [2, 0, 1, 4, 3]), _solved_path(graph, [4, 3, 2, 1, 0])]
    assert len(PathDataSource(graph, paths, size=64).generate()) == 64
    all_states = sum(len(path.edges) + 1 for path in paths)
    assert len(PathDataSource(graph, paths).generate()) == all_states


def test_path_data_source_rejects_invalid_paths():
    graph = _lrx5()
    with pytest.raises(ValueError, match="At least one path is needed"):
        PathDataSource(graph, [])
    with pytest.raises(ValueError, match="does not end at the central state"):
        PathDataSource(graph, [CayleyPath(torch.tensor([2, 0, 1, 4, 3]), [0, 1], graph.definition)])
    with pytest.raises(ValueError, match="Path uses generator 5, but this graph has 3 generators"):
        PathDataSource(graph, [CayleyPath(graph.central_state, [5], graph.definition)])
    with pytest.raises(ValueError, match="n_outputs must be either 1 or the number of generators"):
        PathDataSource(graph, [_solved_path(graph, [2, 0, 1, 4, 3])], n_outputs=2)
    with pytest.raises(ValueError, match="weight must be positive"):
        PathDataSource(graph, [_solved_path(graph, [2, 0, 1, 4, 3])], weight=0)


def _write_tsv(path, rows: list[str]) -> None:
    """Writes a file with the columns of the "cayleypy-beam-results" tables that matter here."""
    header = "puzzle_id\tsolution_length\tsolution\tauthor_name"
    path.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")


def test_path_data_source_from_tsv(tmp_path):
    graph = _lrx5()
    path = _solved_path(graph, [2, 0, 1, 4, 3])
    solution = graph.definition.path_to_string(path.edges)
    file = tmp_path / "solutions.tsv"
    # Solutions in those tables are prefixed with an apostrophe, so that spreadsheets keep them as text.
    _write_tsv(file, [f"7\t{len(path.edges)}\t'{solution}\tsomebody"])

    data = PathDataSource.from_tsv(file, graph).generate()
    # The state that was solved is not in the file - it is restored by replaying the solution backwards.
    assert torch.equal(data.states[0], path.start_state)
    assert float(data.targets[0]) == len(path.edges)
    assert torch.equal(data.states[-1], graph.central_state)
    assert float(data.targets[-1]) == 0


def test_path_data_source_from_tsv_reads_only_requested_puzzle(tmp_path):
    graph = _lrx5()
    short = _solved_path(graph, [1, 0, 2, 3, 4])
    long = _solved_path(graph, [2, 0, 1, 4, 3])
    file = tmp_path / "solutions.tsv"
    _write_tsv(
        file,
        [
            f"7\t{len(short.edges)}\t{graph.definition.path_to_string(short.edges)}\tsomebody",
            f"8\t{len(long.edges)}\t{graph.definition.path_to_string(long.edges)}\tsomebody else",
        ],
    )

    assert len(PathDataSource.from_tsv(file, graph).generate()) == len(short.edges) + len(long.edges) + 2
    for puzzle_id, path in [("7", short), ("8", long)]:
        data = PathDataSource.from_tsv(file, graph, puzzle_id=puzzle_id).generate()
        assert len(data) == len(path.edges) + 1
        assert torch.equal(data.states[0], path.start_state)


def test_path_data_source_from_tsv_rejects_bad_files(tmp_path):
    graph = _lrx5()
    file = tmp_path / "solutions.tsv"

    _write_tsv(file, ["7\t2\tL.Z\tsomebody"])
    with pytest.raises(ValueError, match='uses move "Z", which is not a generator'):
        PathDataSource.from_tsv(file, graph)

    _write_tsv(file, ["7\t1\tL\tsomebody"])
    with pytest.raises(ValueError, match='File has no column "moves"'):
        PathDataSource.from_tsv(file, graph, column="moves")

    _write_tsv(file, [])
    with pytest.raises(ValueError, match="No solutions were read"):
        PathDataSource.from_tsv(file, graph)

    _write_tsv(file, ["7\t1\tL\tsomebody"])
    with pytest.raises(ValueError, match="No solutions were read"):
        PathDataSource.from_tsv(file, graph, puzzle_id="42")


def test_path_data_source_from_tsv_rejects_graph_that_is_not_inverse_closed(tmp_path):
    graph = CayleyGraph(PermutationGroups.lx(5), device="cpu")
    file = tmp_path / "solutions.tsv"
    _write_tsv(file, ["7\t1\tL\tsomebody"])
    with pytest.raises(ValueError, match="not inverse closed"):
        PathDataSource.from_tsv(file, graph)


def test_mixture_keeps_requested_proportions():
    torch.manual_seed(0)
    data = MixtureDataSource([_ConstantSource(1000, 1.0), _ConstantSource(100, 2.0)], [0.9, 0.1]).generate()
    # The second source has only 100 states, so it defines the total: 100 is 10% of 1000.
    assert len(data) == 1000
    assert int((data.targets == 1.0).sum()) == 900
    assert int((data.targets == 2.0).sum()) == 100

    equal_shares = MixtureDataSource([_ConstantSource(50, 1.0), _ConstantSource(50, 2.0)]).generate()
    assert int((equal_shares.targets == 1.0).sum()) == 50
    assert int((equal_shares.targets == 2.0).sum()) == 50


def test_mixture_does_not_repeat_states_of_a_source_that_generated_too_little():
    torch.manual_seed(0)
    # 10 states cannot be 50% of anything larger than 20 states without repeating them.
    data = MixtureDataSource([_ConstantSource(1000, 1.0), _ConstantSource(10, 2.0)], [0.5, 0.5]).generate()
    assert len(data) == 20
    assert int((data.targets == 1.0).sum()) == 10
    assert int((data.targets == 2.0).sum()) == 10


def test_mixture_never_drops_a_source_completely():
    torch.manual_seed(0)
    data = MixtureDataSource([_ConstantSource(100, 1.0), _ConstantSource(1, 2.0)], [0.999, 0.001]).generate()
    assert int((data.targets == 2.0).sum()) == 1


def test_mixture_of_sparse_and_dense_targets():
    torch.manual_seed(0)
    graph = _lrx5()
    walks = SparseQSampler(graph, n_walks=8, rw_length=5)
    anchors = BfsAnchors(graph, depth=2, size=10, n_outputs=3)
    data = MixtureDataSource([walks, anchors], [0.8, 0.2]).generate()

    assert data.n_outputs == 3
    assert data.mask is not None
    # 40 states from the walks are 80% of 50 states, and 10 anchors are the remaining 20%.
    assert len(data) == 50
    # Anchors know all children exactly, so their rows are fully labeled, while walks label at most 2 moves.
    n_labeled = data.mask.sum(dim=1)
    assert int((n_labeled == 3).sum()) == 10
    assert int((n_labeled <= 2).sum()) == 40


def test_mixture_rejects_invalid_values():
    source = _ConstantSource(10, 1.0)
    with pytest.raises(ValueError, match="At least one data source is needed"):
        MixtureDataSource([])
    with pytest.raises(ValueError, match="There are 2 sources, but 3 fractions for them"):
        MixtureDataSource([source, source], [0.5, 0.25, 0.25])
    with pytest.raises(ValueError, match="Fractions must sum to 1"):
        MixtureDataSource([source, source], [0.5, 0.4])
    with pytest.raises(ValueError, match="fraction must be positive"):
        MixtureDataSource([source, source], [1.5, -0.5])
