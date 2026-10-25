"""The den-reg task for the Mamba-2 den-reg."""

from train.experimental.den_reg.task import DenoisedRegressionWithDenoiserMetrics


class MambaDenoisedRegression(DenoisedRegressionWithDenoiserMetrics):
    """``DenoisedRegressionWithDenoiserMetrics`` whose state-space parameter
    group (learning rate ``ssm_lr``, no weight decay) holds Mamba-2's A_log
    and dt_bias, in the role S4D's log_dt and A hold in the S4D den-reg."""

    SSM_PARAM_NAMES = ("A_log", "dt_bias")
