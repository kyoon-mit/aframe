from .autoencoder import AutoencoderAframe
from .base import AframeBase, ClassificationAframe
from .regression import LitLinOSSGaussianNLL, LitS4DGaussianNLL, RegressionAframe
from .supervised import (
    SupervisedAframe,
    SupervisedAframeS4,
    SupervisedMultiModalAframe,
    SupervisedTimeSpectrogramAframe,
)
