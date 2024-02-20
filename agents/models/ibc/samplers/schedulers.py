#
# Classes are used from: https://github.com/google-research/ibc/blob/master/ibc/agents/mcmc.py#L267


class PolynomialSchedule:
    """
    Polynomial learning rate schedule for Langevin sampler.
    """

    def __init__(self, init, final, power, num_steps):
        self._init = init
        self._final = final
        self._power = power
        self._num_steps = num_steps

    def get_rate(self, index):
        """
        Get learning rate for index.
        """
        return (
            (self._init - self._final)
            * ((1 - (float(index) / float(self._num_steps - 1))) ** (self._power))
        ) + self._final


class ExponentialSchedule:
    """
    Exponential learning rate schedule for Langevin sampler.
    """

    def __init__(self, init, decay):
        self._decay = decay
        self._latest_lr = init
        self.min_lr = 1e-5

    def get_rate(self, index):
        """Get learning rate. Assumes calling sequentially."""
        del index
        self._latest_lr *= self._decay
        if self._latest_lr < self.min_lr:
            return self.min_lr
        else:
            return self._latest_lr
