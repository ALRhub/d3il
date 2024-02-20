import logging
import abc
import os

log = logging.getLogger(__name__)


class BaseSim(abc.ABC):

    def __init__(
            self,
            seed: int,
            device: str,
            render: bool = True,
            n_cores: int = 1,
            if_vision: bool = False
    ):
        self.seed = seed
        self.device = device
        self.render = render
        self.n_cores = n_cores

        self.if_vision = if_vision

        self.working_dir = os.getcwd()
        self.env_name = 'BaseEnvironment'

    @abc.abstractmethod
    def test_agent(self, agent):
        pass
