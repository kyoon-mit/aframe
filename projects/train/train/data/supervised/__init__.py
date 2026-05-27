from .time_frequency_domain import (
    FrequencyDomainSupervisedAframeDataset,
    SpectrogramDomainSupervisedAframeDataset,
    TimeSpectrogramDomainSupervisedAframeDataset,
)
from .multimodal import MultiModalSupervisedAframeDataset
from .supervised import SupervisedAframeDataset

from .time_domain import (
    CurriculumPowerLaw,
    TimeDomainSupervisedAframeDataset,
    HeterodyneTimeDomainSupervisedAframeDataset,
)
from .mixin import RegressionSupervisedMixin


class TimeDomainRegressionDataset(
    RegressionSupervisedMixin, TimeDomainSupervisedAframeDataset
):
    pass
