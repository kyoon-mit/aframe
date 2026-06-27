from .base import Architecture
from .regression import (
    MultiTaskArchitecture,
    MultiTaskTimeDomainResNet,
    RegressionArchitecture,
    RegressionTimeDomainResNet,
    RegressionTimeDomainLinOSS,
    RegressionTimeDomainOriginalLinOSS,
)
from .supervised import (
    SupervisedArchitecture,
    SupervisedFrequencyDomainResNet,
    SupervisedMultiModalResNet,
    SupervisedSpectrogramDomainResNet,
    SupervisedTimeDomainResNet,
    SupervisedTimeSpectrogramResNet,
    SupervisedHeterodyneTimeDomainResNet,
    SupervisedTimeDomainOriginalLinOSS,
)
