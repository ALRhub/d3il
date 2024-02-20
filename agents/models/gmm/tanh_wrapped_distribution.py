import torch.distributions as D
import torch


class TanhWrappedDistribution(D.Distribution):
    def __init__(self, base_distribution: D.Distribution, scale: float = 1.0):
        super(TanhWrappedDistribution, self).__init__()

        self.base_distribution = base_distribution
        self.tanh_epsilon = 1e-6
        self.scale = scale

    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        z = self.base_distribution.sample(sample_shape=sample_shape).detach()

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        z = self.base_distribution.rsample(sample_shape=sample_shape)

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    def log_prob(self, value: torch.Tensor):
        value = value / self.scale

        one_plus_x = (1 + value).clamp(min=self.tanh_epsilon)
        one_minus_x = (1 - value).clamp(min=self.tanh_epsilon)
        pre_tanh_value = 0.5 * torch.log(one_plus_x) - 0.5 * torch.log(one_minus_x)

        lp = self.base_distribution.log_prob(pre_tanh_value)
        tanh_lp = torch.log(1 - value.pow(2) + self.tanh_epsilon)

        # In case the base dist already sums up the log probs, make sure we do the same
        return lp - tanh_lp if len(lp.shape) == len(tanh_lp.shape) else lp - tanh_lp.sum(-1)