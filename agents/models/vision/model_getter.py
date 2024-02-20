import torch
import torchvision
from typing import List, Callable
from agents.models.robomimic.models.obs_core import VisualCore


def get_resnet(input_shape: List[int], output_size: int):
    """Get ResNet model from torchvision.models
    Args:
        input_shape: Shape of input image (C, H, W).
        output_size: Size of output feature vector.
    """

    resnet = VisualCore(
        input_shape=input_shape,
        backbone_class="ResNet18Conv",
        backbone_kwargs=dict(
            input_coord_conv=False,
            pretrained=False,
        ),
        pool_class="SpatialSoftmax",
        pool_kwargs=dict(
            num_kp=32,
            learnable_temperature=False,
            temperature=1.0,
            noise_std=0.0,
            output_variance=False,
        ),
        flatten=True,
        feature_dimension=output_size,
    )

    return resnet


def _get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    num_fc_in = resnet.fc.in_features

    resnet.fc = torch.nn.Linear(num_fc_in, 64)
    # resnet.fc = torch.nn.Identity()

    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model
