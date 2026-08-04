"""Library of pretrained models."""

from .models import ModelConfig

# Pretrained models to be used as predictors for Beam Search.
PREDICTOR_MODELS = {
    # Q-model: it has one output per generator, so it estimates distances of all children of a state in one forward
    # pass, and beam search must be called with `use_child_scores=True` to use it. Trained with `cayleypy.train`, see
    # section "Predictor models" of README for the training recipe.
    "lrx-14": ModelConfig(
        model_type="RESMLP",
        input_size=14,
        num_classes_for_one_hot=14,
        layers_sizes=[512, 512, 512],
        n_outputs=3,
        graph_hash="d697b93f3ca3b17f695af3e046b3438c71c3521d9a4955ecf78b85e5fd27ec47",
        weights_kaggle_id="rokham/lrx-14-q/pyTorch/resmlp-512x3/1",
        weights_path="lrx_14_q_resmlp.pt",
    ),
    "lrx-16": ModelConfig(
        model_type="MLP",
        input_size=16,
        num_classes_for_one_hot=16,
        layers_sizes=[256, 256, 256],
        weights_kaggle_id="fedimser/lrx-16/pyTorch/ep60/1",
        weights_path="model_ep60.pth",
    ),
    "lrx-32": ModelConfig(
        model_type="MLP",
        input_size=32,
        num_classes_for_one_hot=32,
        layers_sizes=[1024, 1024, 1024],
        weights_kaggle_id="fedimser/lrx-32-by-mrnnnn/PyTorch/model_final/1",
        weights_path="model_final.pth",
    ),
}
