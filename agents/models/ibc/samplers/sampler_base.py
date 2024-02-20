
import numpy as np


class ConditionalEBMSamplerBase:

    def __init__(self) -> None:
        pass

    def get_device(self, device: str):
        self.device = device

    def get_bounds(self, bounds: np.ndarray):
        self.bounds = bounds

    def gen_train_samples(self):
       raise NotImplementedError()
    
    def infer(self):
        raise NotImplementedError()


class MarginalEBMSamplerBase:

    def __init__(self) -> None:
        pass

    def get_device(self, device: str):
        self.device = device

    def get_bounds(self, bounds: np.ndarray):
        self.bounds = bounds

    def gen_train_samples(self):
       raise NotImplementedError()
    
    def infer(self):
        raise NotImplementedError()
