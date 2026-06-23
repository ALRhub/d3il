import logging
import abc
import os

# Rendering must ALWAYS use the EGL offscreen backend (GPU, headless-safe).
# base_sim is imported by every task simulator, so pinning it here guarantees
# the backend is set before any MuJoCo render context is created -- it is never
# auto-selected (the auto order tries GLFW first, which crashes without a
# display) and never falls back to the slow OSMesa CPU renderer.
os.environ["MUJOCO_GL"] = "egl"

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
