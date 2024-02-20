
import torch
import torch.nn as nn


'''
Simple helper class that gets an input batch and returns the state and goal observations
if we do not have a goal a None is returned for the goal variable 
typically used for simple state, action pairs that do not require embeddings or processing
'''
class NoEncoder(nn.Module):
    def __init__(self, device: str, state_modality: str, goal_modality: str):
        super(NoEncoder, self).__init__()
        self.state_modality = state_modality
        self.goal_modality = goal_modality
        self.device = device
    
    @torch.no_grad()
    def forward(self, x: dict):
        state = x[self.state_modality].to(self.device)
        goal = x[self.goal_modality].to(self.device) if self.goal_modality in x else None
        return state, goal 