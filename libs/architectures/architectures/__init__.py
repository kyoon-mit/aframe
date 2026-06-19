from .base import Architecture
from .regression import (
    MultiTaskArchitecture,
    MultiTaskTimeDomainResNet,
    RegressionArchitecture,
    RegressionTimeDomainResNet,
    RegressionTimeDomainLinOSS,
)
from .supervised import (
    SupervisedArchitecture,
    SupervisedFrequencyDomainResNet,
    SupervisedMultiModalResNet,
    SupervisedSpectrogramDomainResNet,
    SupervisedTimeDomainResNet,
    SupervisedTimeSpectrogramResNet,
    SupervisedHeterodyneTimeDomainResNet,
)
