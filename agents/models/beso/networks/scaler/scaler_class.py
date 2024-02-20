
import logging

import numpy as np
import torch 
import einops

log = logging.getLogger(__name__)


class Scaler:
    """
    Scaler, that scales input and output between 0 and 1
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, scale_data: bool, device: str):
        self.scale_data = scale_data
        self.device = device
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.detach().cpu().numpy()
            y_data = y_data.detach().cpu().numpy()
        # check the length and rearrange if required
        if len(x_data.shape) == 2:
            pass
        elif len(x_data.shape) == 3:
            x_data = einops.rearrange(x_data, "s t x -> (s t) x")
            y_data = einops.rearrange(y_data, "s t x -> (s t) x")
        elif len(x_data.shape) == 4:
            pass
        else:
            raise ValueError('not implemented yet!')
        
        self.x_mean = torch.from_numpy(x_data.mean(0)).to(device)
        self.x_std = torch.from_numpy(x_data.std(0)).to(device)
        self.y_mean = torch.from_numpy(y_data.mean(0)).to(device)
        self.y_std = torch.from_numpy(y_data.std(0)).to(device)
        self.x_max = torch.from_numpy(x_data.max(0)).to(device)
        self.x_min = torch.from_numpy(x_data.min(0)).to(device)
        self.y_min = torch.from_numpy(y_data.min(0)).to(device)
        self.y_max = torch.from_numpy(y_data.max(0)).to(device)

        self.y_bounds = np.zeros((2, y_data.shape[-1]))
        self.x_bounds = np.zeros((2, x_data.shape[-1]))
        # if we scale our input data the bounding values for the sampler class 
        # must be also scaled 
        if self.scale_data:
            self.y_bounds[0, :] = (y_data.min(0) - y_data.mean(0)) / (y_data.std(0) + 1e-12 * np.ones((self.y_std.shape)))[:]
            self.y_bounds[1, :] = (y_data.max(0) - y_data.mean(0)) / (y_data.std(0) + 1e-12 * np.ones((self.y_std.shape)))[:]
            self.x_bounds[0, :] = (x_data.min(0) - x_data.mean(0)) / (x_data.std(0) + 1e-12 * np.ones((self.x_std.shape)))[:]
            self.x_bounds[1, :] = (x_data.max(0) - x_data.mean(0)) / (x_data.std(0) + 1e-12 * np.ones((self.x_std.shape)))[:]

            self.y_bounds_tensor = torch.from_numpy(self.y_bounds).to(device)
            self.x_bounds_tensor = torch.from_numpy(self.x_bounds).to(device)

        else:
            self.y_bounds[0, :] = y_data.min(0)
            self.y_bounds[1, :] = y_data.max(0)
            self.x_bounds[0, :] = x_data.min(0)
            self.x_bounds[1, :] = x_data.max(0)

            self.y_bounds_tensor = torch.from_numpy(self.y_bounds).to(device)
            self.x_bounds_tensor = torch.from_numpy(self.x_bounds).to(device)

        log.info('Datset Info: state min: {} and max: {}, action min: {} and max: {}'.format(self.x_bounds[0, :],self.x_bounds[1, :],
                                                                                                 self.y_bounds[0, :],self.y_bounds[1, :]))
        self.tensor_y_bounds = torch.from_numpy(self.y_bounds).to(device)
        
        log.info(f'Training dataset size: input {x_data.shape} target {y_data.shape}')

    @torch.no_grad()
    def scale_input(self, x):
        """
        Scales the input tensor `x` based on the defined scaling parameters.

        Args:
            x (torch.Tensor): The input tensor to be scaled.
        Returns:
            torch.Tensor: The scaled input tensor
        """
        # ugly workaround if else for the 4 dimension block push goal processing
        if x.shape[-1] == 4 and len(self.x_mean) == 16:
            out = self.scale_aligning_goal(x)
            return out 
        # if we use onehot encoding with the kitchen environmetn we dont need scaling
        elif x.shape[-1] == 7 and len(self.x_mean) == 30:
            return x.to(self.device)
        else:
            x = x.to(self.device)
            if self.scale_data:
                out = (x - self.x_mean) / (self.x_std + 1e-12 * torch.ones((self.x_std.shape), device=self.device))
                return out.to(torch.float32)
            else:
                return x.to(self.device)

    @torch.no_grad()
    def scale_output(self, y):
        """
        Scales the output tensor `y` based on the defined scaling parameters.
        Args:
            y (torch.Tensor): The output tensor to be scaled.
        Returns:
            torch.Tensor: The scaled output tensor.
        """
        y = y.to(self.device)
        if self.scale_data:
            out = (y - self.y_mean) / (self.y_std + 1e-12 * torch.ones((self.y_std.shape), device=self.device))
            return out.to(torch.float32)
        else:
            return y.to(self.device)

    @torch.no_grad()
    def inverse_scale_input(self, x):
        """
        Inversely scales the input tensor `x` based on the defined scaling parameters.

        Args:
            x (torch.Tensor): The input tensor to be inversely scaled.
        Returns:
            torch.Tensor: The inversely scaled input tensor.
        """
        if self.scale_data:
            out = x * (self.x_std + 1e-12 * torch.ones((self.x_std.shape), device=self.device)) + self.x_mean
            return out.to(torch.float32)
        else:
            return x.to(self.device)

    @torch.no_grad()
    def inverse_scale_output(self, y):
        """
        Inversely scales the output tensor `y` based on the defined scaling parameters.

        Args:
            y (torch.Tensor): The output tensor to be inversely scaled.
        Returns:
            torch.Tensor: The inversely scaled output tensor.
        """
        if self.scale_data:
            y.to(self.device)
            out = y * (self.y_std + 1e-12 * torch.ones((self.y_std.shape), device=self.device)) + self.y_mean
            return out
        else:
            return y.to(self.device)
    
    @torch.no_grad()
    def scale_aligning_goal(self, x):
        """
        Scales the input tensor `x` specifically for block push goals.

        Args:
            x (torch.Tensor): The input tensor to be scaled.

        Returns:
            torch.Tensor: The scaled input tensor for block push goals.
        """
        if self.scale_data:
            x = x.to(self.device)
            out = x * (x - self.x_mean[[0, 1 , 3, 4]]) / (self.x_std[[0, 1 , 3, 4]] + 1e-12 * torch.ones((self.x_std[[0, 1 , 3, 4]].shape), device=self.device))
            return out
        else:
            return x.to(self.device)
    
    @torch.no_grad()
    def clip_action(self, y):
        """
        Clips the input tensor `y` based on the defined action bounds.
        """
        return torch.clamp(y, self.y_bounds_tensor[0, :]*1.1, self.y_bounds_tensor[1, :]*1.1).to(self.device).to(torch.float32)


class MinMaxScaler:
    """
    Min Max scaler, that scales the output data between -1 and 1 and the input data between 0 and 1.
    """
    def __init__(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        scale_data: bool, 
        device: str
    ):
        self.scale_data = scale_data
        self.device = device
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.detach().cpu().numpy()
            y_data = y_data.detach().cpu().numpy()
        # check the length and rearrange if required
        if len(x_data.shape) == 2:
            pass
        elif len(x_data.shape) == 3:
            x_data = einops.rearrange(x_data, "s t x -> (s t) x")
            y_data = einops.rearrange(y_data, "s t x -> (s t) x")
        elif len(x_data.shape) == 4:
            pass
        else:
            raise ValueError('not implemented yet!')
        
        with torch.no_grad():
            self.x_max = torch.from_numpy(x_data.max(0)).to(device)
            self.x_min = torch.from_numpy(x_data.min(0)).to(device)
            self.y_min = torch.from_numpy(y_data.min(0)).to(device)
            self.y_max = torch.from_numpy(y_data.max(0)).to(device)
            
            self.new_max_x = torch.ones_like(self.x_max)
            self.new_min_x = -1 * torch.ones_like(self.x_max)
            self.new_max_y = torch.ones_like(self.y_max)
            self.new_min_y = -1 * torch.ones_like(self.y_max)

            self.x_mean = torch.from_numpy(x_data.mean(0)).to(device)
            self.x_std = torch.from_numpy(x_data.std(0)).to(device)
            
            self.y_bounds = np.zeros((2, y_data.shape[-1]))
            self.x_bounds = np.zeros((2, x_data.shape[-1]))
            # if we scale our input data the bounding values for the sampler class 
            # must be also scaled 
            if self.scale_data:
                self.y_bounds[0, :] = -1 * np.ones_like(y_data.min(0))[:]
                self.y_bounds[1, :] = np.ones_like(y_data.min(0))[:]
                # self.x_bounds[0, :] = -1 * np.ones_like(x_data.min(0))[:]
                # self.x_bounds[1, :] = np.ones_like(x_data.min(0))[:]
                self.x_bounds[0, :] = (x_data.min(0) - x_data.mean(0)) / (x_data.std(0) + 1e-12 * np.ones((self.x_std.shape)))[:]
                self.x_bounds[1, :] = (x_data.max(0) - x_data.mean(0)) / (x_data.std(0) + 1e-12 * np.ones((self.x_std.shape)))[:]


                self.y_bounds_tensor = torch.from_numpy(self.y_bounds).to(device)
                self.x_bounds_tensor = torch.from_numpy(self.x_bounds).to(device)

            else:
                self.y_bounds[0, :] = y_data.min(0)
                self.y_bounds[1, :] = y_data.max(0)
                self.x_bounds[0, :] = x_data.min(0)
                self.x_bounds[1, :] = x_data.max(0)

                self.y_bounds_tensor = torch.from_numpy(self.y_bounds).to(device)
                self.x_bounds_tensor = torch.from_numpy(self.x_bounds).to(device)

        log.info('Datset Info: state min: {} and max: {}, action min: {} and max: {}'.format(self.x_bounds[0, :],self.x_bounds[1, :],
                                                                                                 self.y_bounds[0, :],self.y_bounds[1, :]))
        self.tensor_y_bounds = torch.from_numpy(self.y_bounds).to(device)
        
        log.info(f'Training dataset size: input {x_data.shape} target {y_data.shape}')

    @torch.no_grad()
    def scale_input(self, x, aligning_goal=False):
        """
        Scales the input tensor `x` based on the defined scaling parameters.

        Args:
            x (torch.Tensor): The input tensor to be scaled.
        Returns:
            torch.Tensor: The scaled input tensor
        """
        # ugly workaround if else for the 4 dimension block push goal processing.
        if x.shape[-1] == 4 and len(self.x_mean) == 16:
            out = self.scale_aligning_goal(x)
            return out 
        # if we use onehot encoding with the kitchen environmetn we dont need scaling
        elif x.shape[-1] == 7 and len(self.x_mean) == 30:
            return x.to(self.device)
        else:
            x = x.to(self.device)
            if self.scale_data:
                out = (x - self.x_mean) / (self.x_std + 1e-12 * torch.ones((self.x_std.shape), device=self.device))
                return out.to(torch.float32)
            else:
                return x.to(self.device)

    @torch.no_grad()
    def scale_output(self, y):
        """
        Scales the output tensor `y` based on the defined scaling parameters.
        Args:
            y (torch.Tensor): The output tensor to be scaled.
        Returns:
            torch.Tensor: The scaled output tensor.
        """
        y = y.to(self.device)
        if self.scale_data:
            out = (y - self.y_min) / (self.y_max - self.y_min ) * (self.new_max_y - self.new_min_y) + self.new_min_y
            return out.to(torch.float32)
        else:
            return y.to(self.device)

    @torch.no_grad()
    def inverse_scale_input(self, x):
        """
        Inversely scales the input tensor `x` based on the defined scaling parameters.

        Args:
            x (torch.Tensor): The input tensor to be inversely scaled.
        Returns:
            torch.Tensor: The inversely scaled input tensor.
        """
        if self.scale_data:
            out = ( x - self.new_min_x) / (self.new_max_x - self.new_min_x) * (self.x_max - self.x_min) + self.x_min
            return out.to(torch.float32)
        else:
            return x.to(self.device)

    @torch.no_grad()
    def inverse_scale_output(self, y):
        """
        Inversely scales the output tensor `y` based on the defined scaling parameters.

        Args:
            y (torch.Tensor): The output tensor to be inversely scaled.
        Returns:
            torch.Tensor: The inversely scaled output tensor.
        """
        if self.scale_data:
            y.to(self.device)
            out = ( y - self.new_min_y) / (self.new_max_y - self.new_min_y) * (self.y_max - self.y_min) + self.y_min
            return out
        else:
            return y.to(self.device)
    
    @torch.no_grad()
    def scale_aligning_goal(self, x):
        """
        Scales the input tensor `x` specifically for block push goals.

        Args:
            x (torch.Tensor): The input tensor to be scaled.

        Returns:
            torch.Tensor: The scaled input tensor for block push goals.
        """
        if self.scale_data:
            x = x.to(self.device)
            out = x * (x - self.x_mean[[0, 1 , 3, 4]]) / (self.x_std[[0, 1 , 3, 4]] + 1e-12 * torch.ones((self.x_std[[0, 1 , 3, 4]].shape), device=self.device))
            return out
        else:
            return x.to(self.device)
    
    @torch.no_grad()
    def clip_action(self, y):
        """
        Clips the input tensor `y` based on the defined action bounds.
        """
        return torch.clamp(y, self.y_bounds_tensor[0, :]*1.1, self.y_bounds_tensor[1, :]*1.1).to(self.device).to(torch.float32)
    