import abc
import os
import logging

import torch
from omegaconf import DictConfig
import hydra

from beso.networks.scaler.scaler_class import Scaler

# A logger for this file
log = logging.getLogger(__name__)


class BaseAgent(abc.ABC):

    def __init__(
        self,
            model: DictConfig,
            input_encoder: DictConfig,
            optimization: DictConfig,
            obs_modalities: list,
            goal_modalities: list,
            target_modality: str,
            device: str,
            max_train_steps: int,
            eval_every_n_steps: int,
            max_epochs: int,
    ):
        self.scaler = None
        self.model = hydra.utils.instantiate(model).to(device)
        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.get_params()
        )
        self.obs_modalities = obs_modalities
        self.goal_modalities = goal_modalities
        self.target_modality = target_modality
        self.input_encoder = hydra.utils.instantiate(input_encoder)
        self.device = device
        self.steps = 0
        self.epochs = max_epochs
        self.max_train_steps = int(max_train_steps)
        self.eval_every_n_steps = eval_every_n_steps
        self.working_dir = os.getcwd()
        total_params = sum(p.numel() for p in self.model.get_params())
        log.info("The model has a total amount of {} parameters".format(total_params))

    @abc.abstractmethod
    def train_agent(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader):
        """
        Main method to train the agent on the given train and test data
        """
        pass

    @abc.abstractmethod
    def train_agent_on_epochs(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, epochs: int):
        """
        Main method to train the agent on the given train and test data
        """
        pass

    @abc.abstractmethod
    def train_agent_on_steps(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, max_train_steps: int):
        """
        Main method to train the agent on the given train and test data
        """
        pass

    @abc.abstractmethod
    def train_step(self, state: torch.Tensor, action: torch.Tensor):
        """
        Executes a single training step on a mini-batch of data
        """
        pass

    @abc.abstractmethod
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        Method for evaluating the model on one batch of data consisting of two tensors
        """
        pass

    @abc.abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        pass

    @abc.abstractmethod
    def set_bounds(self, scaler: Scaler):
        """
        Method to define the required bounds for the sampling classes or agents
        """
        pass

    def get_scaler(self, scaler: Scaler):
        self.scaler = scaler

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """
        if sv_name is None:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        else:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        log.info('Loaded pre-trained model parameters')
    
    @torch.no_grad()
    def process_batch(self, batch: dict, predict: bool = True):
        """
        Processes a batch of data and returns the state, action and goal
        """
        if predict:
            state, goal = self.input_encoder(batch)
            state = self.scaler.scale_input(state)
            goal = self.scaler.scale_input(goal)
            if goal.shape[-1] == 10:
                goal[..., [2, 5, 6, 7, 8, 9]] = 0
            if self.target_modality in batch:
                action = batch[self.target_modality]
                action = self.scaler.scale_output(action)                
                return state, action, goal
            elif 'goal_task_name' in batch:
                goal_task_name = batch['goal_task_name']
                return state, goal, goal_task_name
            else:
                return state, goal, None
                
        else:
            state, goal = self.input_encoder(batch)  
            state = self.scaler.scale_input(state)
            goal = self.scaler.scale_input(goal)
            if goal.shape[-1] == 10:
                    goal[..., [2, 5, 6, 7, 8, 9]] = 0
            if self.target_modality in batch:
                action = batch[self.target_modality]
                action = self.scaler.scale_output(action)
                return state, action, goal
            else:
                return state, goal
    
    def early_stopping(self, best_test_mse, mean_mse, patience, epochs):
        """
        Early stopping method
        """
        if mean_mse < best_test_mse:
            best_test_mse = mean_mse
            self.store_model_weights(self.working_dir)
            self.epochs_no_improvement = 0
        else:
            self.epochs_no_improvement += 1
        if self.epochs_no_improvement > self.patience:
            return True, best_test_mse
        else:
            return False, best_test_mse
        
    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        if sv_name is None:
            torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))
