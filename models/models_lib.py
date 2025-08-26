"""Library of pretrained models."""

from .models import ModelConfig

# Pretrained models to be used as predictors for Beam Search.
PREDICTOR_MODELS = {
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
