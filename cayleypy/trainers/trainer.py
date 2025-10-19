"""Neural network trainer for Cayley graph distance prediction."""

from typing import TYPE_CHECKING, Dict, Any, List
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from ..cayley_graph import CayleyGraph


class Trainer:
    """Trainer for neural network models on Cayley graphs.

    This class provides methods for training neural networks to predict distances
    in Cayley graphs using both supervised learning with random walks and
    reinforcement learning with k-step lookahead.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        graph: "CayleyGraph",
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        cfg: Dict[str, Any],
    ):
        """Initialize the trainer.

        :param model: The neural network model to train.
        :param graph: The Cayley graph to train on.
        :param optimizer: The optimizer for training.
        :param loss_fn: The loss function to use.
        :param cfg: Configuration dictionary containing training parameters.
        """
        self.model = model
        self.graph = graph
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = graph.device
        self.cfg = cfg
        self.nn_success_rates: Dict[Any, Any] = {}

    def train_supervised(
        self,
        model: torch.nn.Module,
        graph: "CayleyGraph",
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        cfg: Dict[str, Any],
    ) -> None:
        """Train the model using supervised learning with random walks.

        This method generates random walks on the Cayley graph and uses them as training data
        to teach the model to predict distances from the start state. The training uses
        non-backtracking random walks for better mixing properties.

        :param model: The neural network model to train.
        :param graph: The Cayley graph to generate walks on.
        :param optimizer: The optimizer for parameter updates.
        :param loss_fn: The loss function to minimize.
        :param cfg: Configuration dictionary with training parameters.
        """
        pbar = tqdm(range(cfg["num_epochs_supervised"]), desc="Training MLP with random walks")
        for epoch in pbar:
            # Generate random walks using non-backtracking mode for better mixing
            x_tr, y_tr = graph.random_walks(
                width=cfg["random_walks_width"], length=cfg["random_walks_length"], mode="nbt", nbt_history_depth=6
            )

            # Convert targets to float and shuffle training data
            y_train = y_tr.float()
            indices = torch.randperm(x_tr.shape[0], dtype=torch.int64, device="cpu")
            x_train = x_tr[indices]
            y_train = y_train[indices]

            model.train()

            # Training loop over batches
            n_states_all = x_train.shape[0]
            cc = 0
            train_loss = 0.0
            for i_start_batch in range(0, n_states_all, cfg["batch_size"]):
                i_end_batch = min(i_start_batch + cfg["batch_size"], n_states_all)

                # Forward pass
                outputs = model(x_train[i_start_batch:i_end_batch])
                loss = loss_fn(outputs.squeeze(), y_train[i_start_batch:i_end_batch])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                cc += 1

            train_loss /= cc

            # Update progress bar with current loss
            if (epoch + 1) % 10 == 0:
                pbar.set_postfix(loss=f"{loss:.4f}")

    def train_rl(
        self,
        model: torch.nn.Module,
        graph: "CayleyGraph",
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        cfg: Dict[str, Any],
        k_steps: int = 2,
    ) -> None:
        """Train the model using reinforcement learning with k-step lookahead.

        This method uses a k-step lookahead approach where the model predicts distances
        for states reached by applying generators k steps ahead. This helps the model
        learn better distance estimates by considering multiple possible paths.

        :param model: The neural network model to train.
        :param graph: The Cayley graph to train on.
        :param optimizer: The optimizer for parameter updates.
        :param loss_fn: The loss function to minimize.
        :param cfg: Configuration dictionary with training parameters.
        :param k_steps: Number of steps to look ahead for RL training.
        """
        tensor_generators = torch.tensor(graph.generators, device=graph.device)
        pbar = tqdm(range(cfg["num_epochs_rl"]), desc=f"{k_steps}-step RL")

        for epoch in pbar:
            # Generate random walks for training data
            x_all, _ = graph.random_walks(
                width=cfg["random_walks_width"], length=cfg["random_walks_length"], mode="nbt", nbt_history_depth=6
            )
            x_all = x_all.to(graph.device)
            n_states = x_all.size(0)
            n_actions = tensor_generators.size(0)
            state_size = x_all.size(1)
            y_all = torch.full((n_states,), float("inf"), device=graph.device)

            model.eval()
            with torch.no_grad():
                # k-step lookahead: for each generator, apply it k times and get predictions
                for action_idx in range(n_actions):
                    gen = tensor_generators[action_idx]
                    neighbors = torch.gather(x_all, 1, gen.expand(n_states, -1))
                    preds = []

                    # Process in batches to avoid memory issues
                    for i in range(0, n_states, 256):
                        batch = neighbors[i : i + 256]
                        rollout = batch
                        path_preds = []

                        # Apply generator k times and collect predictions
                        for step in range(k_steps):
                            pred = model(rollout).squeeze()
                            path_preds.append(pred + step + 1)
                            rollout = torch.gather(rollout, 1, gen.expand(batch.size(0), -1))

                        # Take minimum prediction across k steps
                        best_pred = torch.min(torch.stack(path_preds, dim=0), dim=0).values
                        preds.append(best_pred)

                    pred = torch.cat(preds, dim=0)
                    y_all = torch.min(y_all, pred)

            # Set goal state distance to 0 and ensure minimum distance of 1
            goal_state = torch.arange(state_size, device=graph.device)
            is_goal = (x_all == goal_state).all(dim=1)
            y_all = torch.clamp_min(y_all, 1)
            y_all[is_goal] = 0
            y_all = y_all.float()

            # Split data into train/validation sets
            n_samples = x_all.size(0)
            n_val = int(n_samples * 0.1)
            n_train = n_samples - n_val
            perm = torch.randperm(n_samples, device=graph.device)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            # Training loop
            model.train()
            total_train_loss = 0.0
            for start in range(0, n_train, cfg["batch_size"]):
                end = start + cfg["batch_size"]
                batch_idx = train_idx[start:end]
                xb = x_all[batch_idx]
                yb = y_all[batch_idx]

                optimizer.zero_grad()
                pred = model(xb).squeeze()
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * xb.size(0)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(x_all[val_idx]).squeeze()
                val_loss = loss_fn(val_pred, y_all[val_idx]).item()

            # Update progress bar
            if (epoch + 1) % 10 == 0:
                anchor_loss = total_train_loss / n_train
                hinge_loss = val_loss
                total_loss = anchor_loss + hinge_loss
                pbar.set_postfix(anchor=f"{anchor_loss:.4f}", hinge=f"{hinge_loss:.4f}", loss=f"{total_loss:.4f}")

    def evaluate(
        self, model: torch.nn.Module, graph: "CayleyGraph", validation_states: List[Any], cfg: Dict[str, Any]
    ) -> None:
        """Evaluate the model using beam search on validation states.

        This method runs beam search with the trained model as a predictor to evaluate
        how well the model can guide the search to find optimal paths.

        :param model: The trained neural network model to use as predictor.
        :param graph: The Cayley graph to perform beam search on.
        :param validation_states: List of states to evaluate the model on.
        :param cfg: Configuration dictionary with evaluation parameters.
        """
        print("Starting beam search evaluation...")
        for start_state in validation_states:
            self.nn_success_rates[start_state] = graph.beam_search(
                start_state=start_state,
                beam_width=1,
                beam_mode=cfg["beam_mode"],
                max_steps=cfg["max_steps"],
                predictor=model,
                history_depth=cfg["history_depth"],
                verbose=1,
            )

    def train(self, validation_states: List[Any]) -> None:
        """Run the complete training and evaluation pipeline.

        This method orchestrates the full training process including:
        1. Supervised training with random walks
        2. Initial evaluation with beam search
        3. Optional reinforcement learning fine-tuning
        4. Final evaluation

        :param validation_states: List of states to evaluate the model on.
        """
        print("\n=== Starting model training and evaluation process ===\n")

        print("\n3. Setting up loss function and optimizer...")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["learning_rate"], weight_decay=1e-5)

        print("\n4. Starting supervised training with random walks...")
        self.train_supervised(self.model, self.graph, optimizer, loss_fn, self.cfg)

        print("\n5. Running initial beam search evaluation...")
        self.evaluate(self.model, self.graph, validation_states, self.cfg)

        if self.cfg["num_epochs_rl"] > 0:
            print("\n6. Starting reinforcement learning fine-tuning...")
            self.train_rl(self.model, self.graph, optimizer, loss_fn, self.cfg)

            print("\n7. Running final beam search evaluation...")
            self.evaluate(self.model, self.graph, validation_states, self.cfg)
