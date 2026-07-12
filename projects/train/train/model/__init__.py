from .autoencoder import AutoencoderAframe
from .base import AframeBase
from .classification import AframeClassification
from .multitask import SupervisedMultiTaskAframe
from .regression import GaussianNLLRegressionAframe, SupervisedRegressionAframe
from .regression_ky import (
    GaussianNLLRegressionAframeCustomLR,
    WarmupCosineAnnealingWarmRestarts,
)
from .supervised import (
    SupervisedAframe,
    SupervisedAframeS4,
    SupervisedMultiModalAframe,
    SupervisedTimeSpectrogramAframe,
)
