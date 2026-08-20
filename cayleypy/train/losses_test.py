import pytest
import torch

from .losses import Loss, MseLoss, PinballLoss, make_loss


def test_mse_matches_manual_computation():
    loss = MseLoss()
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 2.0, 5.0])
    assert torch.allclose(loss.elementwise(predictions, targets), torch.tensor([0.25, 0.0, 4.0]))
    assert torch.allclose(loss(predictions, targets), torch.tensor(4.25 / 3))


def test_mse_matches_torch():
    torch.manual_seed(0)
    predictions = torch.randn(20)
    targets = torch.randn(20)
    expected = torch.nn.functional.mse_loss(predictions, targets)
    assert torch.allclose(MseLoss()(predictions, targets), expected)


def test_pinball_at_half_is_half_of_mae():
    torch.manual_seed(0)
    predictions = torch.randn(20)
    targets = torch.randn(20)
    expected = 0.5 * torch.nn.functional.l1_loss(predictions, targets)
    assert torch.allclose(PinballLoss(0.5)(predictions, targets), expected)


def test_pinball_matches_manual_computation():
    # Underestimated by 1, exact, overestimated by 2.
    predictions = torch.tensor([1.0, 2.0, 5.0])
    targets = torch.tensor([2.0, 2.0, 3.0])
    loss = PinballLoss(0.8)
    expected = torch.tensor([0.8 * 1.0, 0.0, 0.2 * 2.0])
    assert torch.allclose(loss.elementwise(predictions, targets), expected)
    assert torch.allclose(loss(predictions, targets), expected.mean())


def test_pinball_penalizes_underestimation_more_when_tau_is_large():
    loss = PinballLoss(0.9)
    target = torch.tensor([10.0])
    underestimate = loss(torch.tensor([9.0]), target)
    overestimate = loss(torch.tensor([11.0]), target)
    assert torch.allclose(underestimate, torch.tensor(0.9))
    assert torch.allclose(overestimate, torch.tensor(0.1))
    # And the other way round for a small tau.
    loss = PinballLoss(0.1)
    assert torch.allclose(loss(torch.tensor([9.0]), target), torch.tensor(0.1))
    assert torch.allclose(loss(torch.tensor([11.0]), target), torch.tensor(0.9))


def test_pinball_is_minimized_at_the_quantile():
    loss = PinballLoss(0.75)
    targets = torch.arange(1, 11).float()
    candidates = torch.arange(1, 11).float()
    losses = torch.tensor([float(loss(c.expand(10), targets)) for c in candidates])
    # 0.75-quantile of 1..10 is 8 (it minimizes the pinball loss uniquely: 9.25 vs 9.75 for 7 and 9).
    assert float(candidates[int(torch.argmin(losses))]) == 8.0


def test_mask_ignores_unlabeled_elements():
    loss = MseLoss()
    predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[2.0, 1e9], [-1e9, 6.0]])
    mask = torch.tensor([[True, False], [False, True]])
    # Only errors of the labeled elements: (1-2)^2 = 1 and (4-6)^2 = 4.
    assert torch.allclose(loss(predictions, targets, mask=mask), torch.tensor(2.5))


def test_mask_can_be_numeric():
    loss = MseLoss()
    predictions = torch.tensor([[1.0, 2.0]])
    targets = torch.tensor([[2.0, 100.0]])
    bool_mask = torch.tensor([[True, False]])
    numeric_mask = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(loss(predictions, targets, mask=numeric_mask), loss(predictions, targets, mask=bool_mask))


@pytest.mark.parametrize("bad_target", [float("nan"), float("inf"), float("-inf")])
def test_mask_ignores_non_finite_targets_of_unlabeled_elements(bad_target):
    """Targets of unlabeled elements are documented to be arbitrary, and that includes non-finite ones."""
    loss = MseLoss()
    predictions = torch.zeros((2, 2), requires_grad=True)
    targets = torch.tensor([[1.0, bad_target], [2.0, 3.0]])
    mask = torch.tensor([[True, False], [True, True]])
    value = loss(predictions, targets, mask=mask)
    # Only the errors of the labeled elements: 1, 4 and 9, over the 3 of them.
    assert torch.allclose(value, torch.tensor(14.0 / 3))
    value.backward()
    assert predictions.grad is not None
    assert not bool(torch.isnan(predictions.grad).any())
    assert float(predictions.grad[0, 1]) == 0.0


def test_mask_magnitude_does_not_reweight_labeled_elements():
    """Nonzero in a mask means "labeled" and nothing else, so a 2 must not count an element twice."""
    loss = MseLoss()
    predictions = torch.zeros(2)
    targets = torch.tensor([1.0, 3.0])
    indicator = loss(predictions, targets, mask=torch.tensor([1.0, 1.0]))
    assert torch.allclose(loss(predictions, targets, mask=torch.tensor([1.0, 2.0])), indicator)


def test_fully_masked_batch_gives_zero_loss_in_half_precision():
    """The lower bound on the denominator has to be representable in the dtype of the predictions."""
    loss = MseLoss()
    predictions = torch.zeros((1, 2), dtype=torch.float16)
    targets = torch.ones((1, 2), dtype=torch.float16)
    assert float(loss(predictions, targets, mask=torch.zeros((1, 2), dtype=torch.float16))) == 0.0


def test_mask_blocks_gradient():
    predictions = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    targets = torch.tensor([[2.0, 1e9, 1e9]])
    mask = torch.tensor([[True, False, False]])
    MseLoss()(predictions, targets, mask=mask).backward()
    assert predictions.grad is not None
    assert float(predictions.grad[0, 0]) != 0.0
    assert float(predictions.grad[0, 1]) == 0.0
    assert float(predictions.grad[0, 2]) == 0.0


def test_fully_masked_batch_gives_zero_loss_and_zero_gradient():
    for loss in [MseLoss(), PinballLoss(0.7)]:
        predictions = torch.tensor([[1.0, 2.0]], requires_grad=True)
        targets = torch.tensor([[5.0, 5.0]])
        value = loss(predictions, targets, mask=torch.zeros((1, 2)))
        assert float(value.detach()) == 0.0
        value.backward()
        assert predictions.grad is not None
        assert not bool(torch.isnan(predictions.grad).any())
        assert torch.equal(predictions.grad, torch.zeros((1, 2)))


def test_weights_give_weighted_mean():
    loss = MseLoss()
    predictions = torch.tensor([1.0, 2.0])
    targets = torch.tensor([2.0, 4.0])
    # Errors are 1 and 4; weighted mean with weights 3 and 1 is (3*1 + 1*4)/4.
    weights = torch.tensor([3.0, 1.0])
    assert torch.allclose(loss(predictions, targets, weights=weights), torch.tensor(7.0 / 4))
    # Weights are not normalized, but scaling all of them does not change the mean.
    assert torch.allclose(loss(predictions, targets, weights=weights * 10), torch.tensor(7.0 / 4))


def test_weights_combine_with_mask():
    loss = MseLoss()
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([2.0, 4.0, 1e9])
    mask = torch.tensor([True, True, False])
    weights = torch.tensor([3.0, 1.0, 100.0])
    assert torch.allclose(loss(predictions, targets, mask=mask, weights=weights), torch.tensor(7.0 / 4))


def test_works_on_q_shaped_predictions():
    loss = MseLoss()
    predictions = torch.zeros((7, 3))
    targets = torch.ones((7, 3))
    value = loss(predictions, targets)
    assert value.shape == torch.Size([])
    assert float(value) == 1.0


def test_accepts_integer_targets():
    # Exact distances computed by BFS are integers.
    predictions = torch.tensor([1.0, 2.0])
    targets = torch.tensor([2, 2], dtype=torch.int64)
    assert torch.allclose(MseLoss()(predictions, targets), torch.tensor(0.5))
    assert torch.allclose(PinballLoss(0.5)(predictions, targets), torch.tensor(0.25))


def test_make_loss():
    assert isinstance(make_loss("mse"), MseLoss)
    pinball = make_loss("pinball", tau=0.9)
    assert isinstance(pinball, PinballLoss)
    assert pinball.tau == 0.9
    assert make_loss("pinball").tau == 0.5  # type: ignore[attr-defined]
    predictions, targets = torch.tensor([1.0]), torch.tensor([2.0])
    assert torch.equal(make_loss("mse")(predictions, targets), MseLoss()(predictions, targets))


def test_training_reduces_loss():
    for loss in [MseLoss(), PinballLoss(0.5), PinballLoss(0.9)]:
        torch.manual_seed(42)
        prediction = torch.zeros(1, requires_grad=True)
        targets = torch.tensor([3.0, 3.0, 3.0])
        optimizer = torch.optim.SGD([prediction], lr=0.1)
        initial_loss = float(loss(prediction.expand(3), targets).detach())
        for _ in range(100):
            optimizer.zero_grad()
            loss(prediction.expand(3), targets).backward()
            optimizer.step()
        final_loss = float(loss(prediction.expand(3), targets).detach())
        assert final_loss < 0.2 * initial_loss
        assert abs(float(prediction.detach()) - 3.0) < 0.1


def test_masked_training_updates_only_labeled_outputs():
    # Emulates training a Q-model where only one output per state is labeled.
    torch.manual_seed(42)
    predictions = torch.zeros((1, 2), requires_grad=True)
    targets = torch.tensor([[5.0, 5.0]])
    mask = torch.tensor([[True, False]])
    optimizer = torch.optim.SGD([predictions], lr=0.1)
    for _ in range(100):
        optimizer.zero_grad()
        MseLoss()(predictions, targets, mask=mask).backward()
        optimizer.step()
    assert abs(float(predictions[0, 0].detach()) - 5.0) < 0.1
    assert float(predictions[0, 1].detach()) == 0.0


def test_loss_is_abstract():
    with pytest.raises(TypeError):
        Loss()  # type: ignore[abstract]  # pylint: disable=abstract-class-instantiated


def test_pinball_rejects_tau_outside_unit_interval():
    for tau in [0.0, 1.0, -0.5, 1.5, float("nan")]:
        with pytest.raises(ValueError, match="tau must be strictly between 0 and 1"):
            PinballLoss(tau)


def test_make_loss_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown loss"):
        make_loss("huber")


def test_rejects_targets_of_wrong_shape():
    with pytest.raises(ValueError, match=r"Shape of targets is \(3,\)"):
        MseLoss()(torch.zeros(2), torch.zeros(3))
    with pytest.raises(ValueError, match="same as shape of predictions"):
        # Same number of elements, but different shape - broadcasting would silently give a wrong answer.
        PinballLoss()(torch.zeros((2, 3)), torch.zeros((3, 2)))


def test_rejects_mask_of_wrong_shape():
    with pytest.raises(ValueError, match=r"Shape of mask is \(2, 1\)"):
        MseLoss()(torch.zeros((2, 3)), torch.zeros((2, 3)), mask=torch.ones((2, 1)))


def test_rejects_weights_of_wrong_shape():
    with pytest.raises(ValueError, match=r"Shape of weights is \(2,\)"):
        MseLoss()(torch.zeros((2, 3)), torch.zeros((2, 3)), weights=torch.ones(2))
