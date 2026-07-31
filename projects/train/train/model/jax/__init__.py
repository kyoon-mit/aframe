from train.model.jax.classification import JaxClassificationAframe
from train.model.jax.denoise_classification import (
    JaxDenoiseClassificationAframe,
)
from train.model.jax.regression import JaxRegressionAframe

__all__ = [
    "JaxRegressionAframe",
    "JaxClassificationAframe",
    "JaxDenoiseClassificationAframe",
]
