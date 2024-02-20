
import abc



class BaseEncoder(abc.ABC):
    
    def __init__(
        self,
        device: str,
    ):
        self.device = device
    
    @abc.abstractmethod
    def forward(self, x: dict):
        pass

    