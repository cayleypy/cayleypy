"""Tests for the Trainer class."""

from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from .trainer import Trainer
from ..cayley_graph import CayleyGraph


class TestTrainer:
    """Test cases for the Trainer class."""

    @pytest.fixture
    def mock_graph(self):
        """Create a mock Cayley graph for testing."""
        graph = Mock(spec=CayleyGraph)
        graph.device = torch.device("cpu")
        graph.generators = [[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]]
        graph.beam_search = Mock(return_value={"success": True, "path_length": 5})
        return graph

    @pytest.fixture
    def simple_model(self):
        """Create a simple neural network model for testing."""
        return nn.Linear(4, 1)

    @pytest.fixture
    def config(self):
        """Create a configuration dictionary for testing."""
        return {
            "num_epochs_supervised": 2,
            "num_epochs_rl": 1,
            "random_walks_width": 100,
            "random_walks_length": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "beam_mode": "greedy",
            "max_steps": 20,
            "history_depth": 3,
        }

    @pytest.fixture
    def trainer(self, simple_model, mock_graph, config):
        """Create a Trainer instance for testing."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=config["learning_rate"])
        loss_fn = nn.MSELoss()
        return Trainer(simple_model, mock_graph, optimizer, loss_fn, config)

    def test_trainer_initialization(self, trainer, simple_model, mock_graph, config):
        """Test that Trainer initializes correctly."""
        assert trainer.model == simple_model
        assert trainer.graph == mock_graph
        assert trainer.CFG == config
        assert trainer.device == mock_graph.device
        assert isinstance(trainer.nn_success_rates, dict)

    def test_trainer_initialization_with_custom_optimizer(self, simple_model, mock_graph, config):
        """Test Trainer initialization with custom optimizer."""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.L1Loss()
        trainer = Trainer(simple_model, mock_graph, optimizer, loss_fn, config)

        assert trainer.model == simple_model
        assert trainer.optimizer == optimizer
        assert trainer.loss_fn == loss_fn

    @patch("cayleypy.trainers.trainer.tqdm")
    def test_train_supervised(self, mock_tqdm, trainer, mock_graph):
        """Test supervised training method."""
        # Mock random_walks to return dummy data
        mock_graph.random_walks.return_value = (torch.randn(100, 4), torch.randint(0, 10, (100,)))  # X_tr  # y_tr

        # Mock tqdm progress bar to be iterable
        mock_pbar = Mock()
        mock_pbar.__iter__ = Mock(return_value=iter(range(trainer.CFG["num_epochs_supervised"])))
        mock_tqdm.return_value = mock_pbar

        # Call train_supervised
        trainer.train_supervised(trainer.model, mock_graph, trainer.optimizer, trainer.loss_fn, trainer.CFG)

        # Verify random_walks was called with correct parameters
        expected_calls = trainer.CFG["num_epochs_supervised"]
        assert mock_graph.random_walks.call_count == expected_calls

        # Check that all calls had the correct parameters
        for call in mock_graph.random_walks.call_args_list:
            assert call.kwargs["width"] == trainer.CFG["random_walks_width"]
            assert call.kwargs["length"] == trainer.CFG["random_walks_length"]
            assert call.kwargs["mode"] == "nbt"
            assert call.kwargs["nbt_history_depth"] == 6

        # Verify tqdm was called
        mock_tqdm.assert_called_once()
        # set_postfix is only called every 10 epochs, so with 2 epochs it won't be called
        # mock_pbar.set_postfix.assert_called()

    @patch("cayleypy.trainers.trainer.tqdm")
    def test_train_rl(self, mock_tqdm, trainer, mock_graph):
        """Test reinforcement learning training method."""
        # Mock random_walks to return dummy data
        mock_graph.random_walks.return_value = (
            torch.randn(100, 4),  # X_all
            torch.randint(0, 10, (100,)),  # y_all (not used in RL)
        )

        # Mock tqdm progress bar to be iterable
        mock_pbar = Mock()
        mock_pbar.__iter__ = Mock(return_value=iter(range(trainer.CFG["num_epochs_rl"])))
        mock_tqdm.return_value = mock_pbar

        # Call train_rl
        trainer.train_rl(trainer.model, mock_graph, trainer.optimizer, trainer.loss_fn, trainer.CFG, k_steps=2)

        # Verify random_walks was called
        mock_graph.random_walks.assert_called_once_with(
            width=trainer.CFG["random_walks_width"],
            length=trainer.CFG["random_walks_length"],
            mode="nbt",
            nbt_history_depth=6,
        )

        # Verify tqdm was called with correct description
        mock_tqdm.assert_called_once()
        call_args = mock_tqdm.call_args
        assert "2-step RL" in str(call_args)

    def test_evaluate(self, trainer, mock_graph):
        """Test evaluation method."""
        validation_states = [torch.tensor([0, 1, 2, 3]), torch.tensor([1, 0, 2, 3])]

        trainer.evaluate(trainer.model, mock_graph, validation_states, trainer.CFG)

        # Verify beam_search was called for each validation state
        assert mock_graph.beam_search.call_count == len(validation_states)

        # Check that success rates were stored
        assert len(trainer.nn_success_rates) == len(validation_states)

    @patch.object(Trainer, "train_supervised")
    @patch.object(Trainer, "evaluate")
    @patch.object(Trainer, "train_rl")
    def test_train_pipeline(self, mock_train_rl, mock_evaluate, mock_train_supervised, trainer):
        """Test the complete training pipeline."""
        validation_states = [torch.tensor([0, 1, 2, 3])]

        # Test with RL enabled
        trainer.CFG["num_epochs_rl"] = 1
        trainer.train(validation_states)

        # Verify all methods were called
        mock_train_supervised.assert_called_once()
        assert mock_evaluate.call_count == 2  # Called twice (before and after RL)
        mock_train_rl.assert_called_once()

    @patch.object(Trainer, "train_supervised")
    @patch.object(Trainer, "evaluate")
    @patch.object(Trainer, "train_rl")
    def test_train_pipeline_no_rl(self, mock_train_rl, mock_evaluate, mock_train_supervised, trainer):
        """Test the training pipeline with RL disabled."""
        validation_states = [torch.tensor([0, 1, 2, 3])]

        # Test with RL disabled
        trainer.CFG["num_epochs_rl"] = 0
        trainer.train(validation_states)

        # Verify supervised training and evaluation were called
        mock_train_supervised.assert_called_once()
        mock_evaluate.assert_called_once()

        # Verify RL was not called
        mock_train_rl.assert_not_called()

    def test_trainer_with_different_loss_functions(self, simple_model, mock_graph, config):
        """Test Trainer with different loss functions."""
        # Test with L1Loss
        l1_loss = nn.L1Loss()
        optimizer = torch.optim.Adam(simple_model.parameters())
        trainer_l1 = Trainer(simple_model, mock_graph, optimizer, l1_loss, config)
        assert trainer_l1.loss_fn == l1_loss

        # Test with SmoothL1Loss
        smooth_l1_loss = nn.SmoothL1Loss()
        trainer_smooth = Trainer(simple_model, mock_graph, optimizer, smooth_l1_loss, config)
        assert trainer_smooth.loss_fn == smooth_l1_loss

    def test_trainer_with_different_optimizers(self, simple_model, mock_graph, config):
        """Test Trainer with different optimizers."""
        # Test with SGD
        sgd_optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        trainer_sgd = Trainer(simple_model, mock_graph, sgd_optimizer, loss_fn, config)
        assert trainer_sgd.optimizer == sgd_optimizer

        # Test with AdamW
        adamw_optimizer = torch.optim.AdamW(simple_model.parameters(), lr=0.001)
        trainer_adamw = Trainer(simple_model, mock_graph, adamw_optimizer, loss_fn, config)
        assert trainer_adamw.optimizer == adamw_optimizer

    def test_trainer_config_validation(self, simple_model, mock_graph):
        """Test that Trainer handles configuration properly."""
        config = {
            "num_epochs_supervised": 5,
            "num_epochs_rl": 3,
            "random_walks_width": 200,
            "random_walks_length": 15,
            "batch_size": 64,
            "learning_rate": 0.0001,
            "beam_mode": "beam",
            "max_steps": 30,
            "history_depth": 5,
        }

        optimizer = torch.optim.Adam(simple_model.parameters())
        loss_fn = nn.MSELoss()
        trainer = Trainer(simple_model, mock_graph, optimizer, loss_fn, config)

        assert trainer.CFG == config
        assert trainer.CFG["num_epochs_supervised"] == 5
        assert trainer.CFG["num_epochs_rl"] == 3
        assert trainer.CFG["batch_size"] == 64

    def test_trainer_device_handling(self, simple_model, mock_graph, config):
        """Test that Trainer properly handles device assignment."""
        # Test with CPU device
        mock_graph.device = torch.device("cpu")
        optimizer = torch.optim.Adam(simple_model.parameters())
        loss_fn = nn.MSELoss()
        trainer = Trainer(simple_model, mock_graph, optimizer, loss_fn, config)

        assert trainer.device == torch.device("cpu")

        # Test with CUDA device (if available)
        if torch.cuda.is_available():
            mock_graph.device = torch.device("cuda")
            trainer_cuda = Trainer(simple_model, mock_graph, optimizer, loss_fn, config)
            assert trainer_cuda.device == torch.device("cuda")

    def test_trainer_success_rates_tracking(self, trainer):
        """Test that Trainer properly tracks success rates."""
        # Initially empty
        assert len(trainer.nn_success_rates) == 0

        # Add some success rates
        trainer.nn_success_rates["state1"] = {"success": True, "path_length": 5}
        trainer.nn_success_rates["state2"] = {"success": False, "path_length": None}

        assert len(trainer.nn_success_rates) == 2
        assert trainer.nn_success_rates["state1"]["success"] is True
        assert trainer.nn_success_rates["state2"]["success"] is False
