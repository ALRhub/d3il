import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Optional

def compute_gradient_loss(
    grad_norm
):
    # clip delta norm values below 0
    # same clipping used in the original ibc implementation
    # clip the delta to max(0, grad_delta)
    grad_delta = grad_norm - 1
    grad_delta = grad_delta.clamp(min=0, max=1e10)
    # get the mean of the delta norm
    wgan_grad_norm = torch.mean(grad_delta.pow(2))

    return wgan_grad_norm

def compute_info_nce_loss(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor, device: str, avrg_e_regularization: float, goal: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the InfoNCE loss of the model based on the prediction of the
    ebm model and the ground truth action located at actions[:, 0, :].
    The InfoNCE loss can be understood as to classify the correct action from
    the set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  actions:   torch.Tensor of the ground truth action and the negative ones
    :param  device:     type of torch device
    :param avrg_e_regularization  float of the regularization value

    :return:            torch.Tensor of the computed InfoNCE loss value
    """
    # add small noise to the positive sample
    actions[:, 0, :] += torch.randn_like(actions[:, 0, :]) * 1e-4
    
    # Generate a random permutation of the positives and negatives.
    permutation = torch.rand(actions.size(0), actions.size(1)).argsort(
        dim=1
    )  # [B, N+1]
    actions = actions[
        torch.arange(actions.size(0)).unsqueeze(-1), permutation
    ]  # [B, N+1, D ]

    # Get the original index of the positive. This will serve as the class label
    # for the loss.
    ground_truth = (permutation == 0).nonzero()[:, 1].to(device)

    # For every element in the mini-batch, there is 1 positive for which the EBM
    # should output a low energy value, and N negatives for which the EBM should
    # output high energy values.
    energy = ebm(state, actions, goal)

    # get the average delta energy for debugging and training performance info
    delta_energy = energy[permutation==0].mean() - energy[permutation!=0].mean()

    # Interpreting the energy as a negative logit, we can apply a cross entropy loss
    # to train the EBM.
    logits = -1.0 * energy
    loss = F.cross_entropy(logits, ground_truth)

    # add regularization loss
    reg_loss = torch.pow(energy, 2).mean()
    # log the average minimum and maximum energy of the current batch for additional debugging and information
    loss_dict = (
        {
            "mean energy": energy.mean().item(),
            "min energy": energy.min().item(),
            "max energy": energy.max().item(),
            "delta energy": delta_energy,
            "avrg_positive_sample_energy": energy[permutation==0].mean(),
            "avrg_negative_sample_energy": energy[permutation!=0].mean(),
            "reg_loss": reg_loss,
            "loss_no_reg": loss
        }
    )
    loss = loss + avrg_e_regularization * reg_loss

    return loss.sum(), loss_dict


def compute_autoregressive_info_nce_loss(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor, device: str, avrg_e_regularization: float
) -> torch.Tensor:
    """
    Computes the InfoNCE loss of the model based on the prediction of the
    ebm model and the ground truth action located at actions[:, 0, :].
    The InfoNCE loss can be understood as to classify the correct action from
    the set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  actions::   torch.Tensor of the ground truth action and the negative ones
    :param  device:     type of torch device
    :param avrg_e_regularization  float of the regularization value

    :return:            torch.Tensor of the computed InfoNCE loss value
    """
    # add small noise to the positive sample
    actions[:, 0, :] += torch.randn_like(actions[:, 0, :]) * 1e-4
    
    # Generate a random permutation of the positives and negatives.
    permutation = torch.rand(actions.size(0), actions.size(1)).argsort(
        dim=1
    )  # [B, N+1]
    actions = actions[
        torch.arange(actions.size(0)).unsqueeze(-1), permutation
    ]  # [B, N+1, D ]

    # Get the original index of the positive. This will serve as the class label
    # for the loss.
    ground_truth = (permutation == 0).nonzero()[:, 1].to(device)

    # For every element in the mini-batch, there is 1 positive for which the EBM
    # should output a low energy value, and N negatives for which the EBM should
    # output higher energy values.\
    loss = 0
    for idx in range(actions.shape[-1]):
        energy = ebm.single_dimension_forward(state, actions[:, :, 0:(idx+1)], train_dimension=idx)
        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        single_dim_loss = F.cross_entropy(logits, ground_truth)
        loss += single_dim_loss

    # add regularization loss if wanted
    reg_loss = torch.pow(energy, 2).mean()
        # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": energy.mean().item(),
            "min energy": energy.min().item(),
            "max energy": energy.max().item(),
            "avrg_positive_sample_energy": energy[permutation==0].mean(),
            "avrg_negative_sample_energy": energy[permutation!=0].mean(),
            "reg_loss": reg_loss,
            "loss_no_reg": loss
        }
    )
    loss = loss + avrg_e_regularization * reg_loss

    return loss

def single_dim_info_nce_loss(
        ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor, device: str, train_dimension: int, avrg_e_regularization: float
) -> torch.Tensor:
    # add small noise to the positive sample
    actions[:, 0, :] += torch.randn_like(actions[:, 0, :]) * 1e-4
    
    # Generate a random permutation of the positives and negatives.
    permutation = torch.rand(actions.size(0), actions.size(1)).argsort(
        dim=1
    )  # [B, N+1]
    actions = actions[
        torch.arange(actions.size(0)).unsqueeze(-1), permutation
    ]  # [B, N+1, D ]

    # Get the original index of the positive. This will serve as the class label
    # for the loss.
    ground_truth = (permutation == 0).nonzero()[:, 1].to(device)

    # For every element in the mini-batch, there is 1 positive for which the EBM
    # should output a low energy value, and N negatives for which the EBM should
    # output higher energy values.\
    energy = ebm.single_dimension_forward(state, actions, train_dimension=train_dimension)
    # Interpreting the energy as a negative logit, we can apply a cross entropy loss
    # to train the EBM.
    logits = -1.0 * energy
    loss = F.cross_entropy(logits, ground_truth)

    # add regularization loss if wanted
    reg_loss = torch.pow(energy, 2).mean()
        # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": energy.mean().item(),
            "min energy": energy.min().item(),
            "max energy": energy.max().item(),
            "avrg_positive_sample_energy": energy[permutation==0].mean(),
            "avrg_negative_sample_energy": energy[permutation!=0].mean(),
            "reg_loss": reg_loss,
            "loss_no_reg": loss
        }
    )
    loss = loss + avrg_e_regularization * reg_loss

    return loss



def contrastive_divergence(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor, avrg_e_regularization: float
) -> torch.Tensor:
    """
    Computes the standard contrastive divergence loss for an ebm given a true action and a
    set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  actions:   torch.Tensor of the ground truth action and the negative ones

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    # add small noise to the positive sample
    actions[:, 0, :] += torch.randn_like(actions[:, 0, :]) * 1e-4
    
    true_action_energy = ebm(state, actions[:, 0, :])
    fake_action_energy = ebm(state, actions[:, 1:, :])

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = (
        torch.pow(true_action_energy, 2).mean()
        + torch.pow(fake_action_energy, 2).mean()
    )

    # add losses together
    loss = cd_loss + avrg_e_regularization * reg_loss

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    # return the loss of the model for optimization
    return loss


def contrastive_divergence_kl(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor, avrg_e_regularization: float, kl_loss_factor: float
) -> torch.Tensor:
    """
    Improved version of the contrastive divergence loss with additional KL based on the
    paper https://arxiv.org/pdf/2012.01316.pdf . The loss consists of the basic contrastive divergence
    term and in addition we approximate the entropy of the loss by using L2 distance of ....

    Code is based on the repo https://github.com/yilundu/improved_contrastive_divergence/blob/master/train.py
    to the paper mentioned above.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  actions:   torch.Tensor of the ground truth action and the negative ones

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    true_action_energy = ebm(state, actions[:, 0, :])
    fake_action_energy = ebm(state, actions[:, 1:, :])

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = (
        torch.pow(true_action_energy, 2).mean()
        + torch.pow(fake_action_energy, 2).mean()
    ).mean()

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    entropy_temperature = 1e-1
    # now compute the KL loss term
    # the KL loss term consists of two parts: the entropy loss part and the sampler energy loss part

    # first we need the gradient w.r.t to the mcmc sample:
    ebm.requires_grad_(False)
    actions_2 = actions.clone()
    actions_2.requires_grad = True
    # compute the energy w.r.t to the state sample
    fake_action_energy_2 = ebm(state, actions_2)

    # set the gradient of the ebm back to previous version
    ebm.requires_grad_(True)

    # compute the kl sampler loss part
    kl_opt_loss = fake_action_energy_2.mean()

    # we approximate the entropy loss term using nearest neighbour of our generated samples
    # therefore  we compute the distance of our generated samples
    nn_dist = torch.norm(
        fake_action_energy_2[:, None, Ellipsis]
        - fake_action_energy_2[:, :, None, Ellipsis],
        p=2,
        dim=-1,
    )
    # we sum it up to estimate the entropy of our samples
    # this loss part forces the network to generate diverse samples with low energy
    kl_entropy_loss = torch.log(nn_dist.min(dim=1)[0]).mean()

    # the -0.3 is based on the same loss implementation of the paper
    # finally we add our loss parts together to the total loss and return it
    loss = cd_loss + kl_entropy_loss - kl_loss_factor * kl_opt_loss + avrg_e_regularization*reg_loss
    return loss


def contrastive_divergence_entropy_approx(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """
    Computes the standard contrastive divergence loss with entropy approximation based 
    on the paper: https://imrss2022.github.io/contributions/ta.pdf
    set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  actions:   torch.Tensor of the ground truth action and the negative ones

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    true_action_energy = ebm(state, actions[:, 0, :])
    fake_action_energy = ebm(state, actions[:, 1:, :])

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = torch.pow(fake_action_energy, 2).mean()
    var_loss = torch.pow(fake_action_energy.mean(), 2)

    # add losses together
    loss = cd_loss + 0.5 * reg_loss - 0.5 * var_loss

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    # return the loss of the model for optimization
    return loss

####################################
# all losses designed to sample the state of an EBM E(x,y) they are the counter part to the upper losses desinged to sample the second values y
# while these could be exchanged we wanted to train models to sample in both directions in one training loop, thus the additional losses are required

def compute_state_info_nce_loss(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor, device: str, avrg_e_regularization: float
) -> torch.Tensor:
    """
    Computes the InfoNCE loss of the model based on the prediction of the
    ebm model and the ground truth action located at actions[:, 0, :].
    The InfoNCE loss can be understood as to classify the correct action from
    the set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  taregets:   torch.Tensor of the ground truth action and the negative ones
    :param  device:     torch.device

    :return:            torch.Tensor of the computed InfoNCE loss value
    """
    # add small noise to the positive sample
    state[:, 0, :] += torch.randn_like(state[:, 0, :]) * 1e-4
    
    # Generate a random permutation of the positives and negatives.
    permutation = torch.rand(state.size(0), state.size(1)).argsort(
        dim=1
    )  # [B, N+1]
    state = state[
        torch.arange(state.size(0)).unsqueeze(-1), permutation
    ]  # [B, N+1, D ]

    # Get the original index of the positive. This will serve as the class label
    # for the loss.
    ground_truth = (permutation == 0).nonzero()[:, 1].to(device)

    # For every element in the mini-batch, there is 1 positive for which the EBM
    # should output a low energy value, and N negatives for which the EBM should
    # output high energy values.
    energy = ebm(state, actions)

    # get the average delta energy for debugging and training performance info
    delta_energy = energy[permutation==0].mean() - energy[permutation!=0].mean()

    # Interpreting the energy as a negative logit, we can apply a cross entropy loss
    # to train the EBM.
    logits = -1.0 * energy
    loss = F.cross_entropy(logits, ground_truth)
    # add regularization loss
    reg_loss = torch.pow(energy, 2).mean()

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": energy.mean().item(),
            "min energy": energy.min().item(),
            "max energy": energy.max().item(),
            "delta energy": delta_energy,
            "avrg_positive_sample_energy": energy[permutation==0].mean(),
            "avrg_negative_sample_energy": energy[permutation!=0].mean(),
            "reg_loss": reg_loss,
            "loss_no_reg": loss
        }
    )
    loss = loss + avrg_e_regularization * reg_loss

    return loss.sum()


def state_contrastive_divergence(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor, avrg_e_regularization: float
) -> torch.Tensor:
    """
    Computes the standard contrastive divergence loss for an ebm given a true action and a
    set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  actions:   torch.Tensor of the ground truth action and the negative ones

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    true_action_energy = ebm(state[:, 0, :], actions)
    fake_action_energy = ebm(state[:, 1:, :], actions)

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = (
        torch.pow(true_action_energy, 2).mean()
        + torch.pow(fake_action_energy, 2).mean()
    )

    # add losses together
    loss = cd_loss + avrg_e_regularization * reg_loss

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    # return the loss of the model for optimization
    return loss


def state_contrastive_divergence_kl(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor, avrg_e_regularization: float, kl_loss_factor: float
) -> torch.Tensor:
    """
    Improved version of the contrastive divergence loss with additional KL based on the
    paper https://arxiv.org/pdf/2012.01316.pdf . The loss consists of the basic contrastive divergence
    term and in addition we approximate the entropy of the loss by using L2 distance of ....

    Code is based on the repo https://github.com/yilundu/improved_contrastive_divergence/blob/master/train.py
    to the paper mentioned above.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  actions:   torch.Tensor of the ground truth action and the negative ones

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    true_action_energy = ebm(state[:, 0, :], actions)
    fake_action_energy = ebm(state[:, 1:, :], actions)

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = (
        torch.pow(true_action_energy, 2).mean()
        + torch.pow(fake_action_energy, 2).mean()
    ).mean()

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    entropy_temperature = 1e-1
    # now compute the KL loss term
    # the KL loss term consists of two parts: the entropy loss part and the sampler energy loss part

    # first we need the gradient w.r.t to the mcmc sample:
    ebm.requires_grad_(False)
    state_2 = state.clone()
    state_2.requires_grad = True
    # compute the energy w.r.t to the state sample
    fake_action_energy_2 = ebm(state_2, actions)

    # set the gradient of the ebm back to previous version
    ebm.requires_grad_(True)

    # compute the kl sampler loss part
    kl_opt_loss = fake_action_energy_2.mean()

    # we approximate the entropy loss term using nearest neighbour of our generated samples
    # therefore  we compute the distance of our generated samples
    nn_dist = torch.norm(
        fake_action_energy_2[:, None, Ellipsis]
        - fake_action_energy_2[:, :, None, Ellipsis],
        p=2,
        dim=-1,
    )
    # we sum it up to estimate the entropy of our samples
    # this loss part forces the network to generate diverse samples with low energy
    kl_entropy_loss = torch.log(nn_dist.min(dim=1)[0]).mean()

    # the -0.3 is based on the same loss implementation of the paper
    # finally we add our loss parts together to the total loss and return it
    loss = cd_loss + kl_entropy_loss - kl_loss_factor * kl_opt_loss + avrg_e_regularization*reg_loss
    return loss

def state_contrastive_divergence_entropy_approx(
    ebm: nn.Module, state: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """
    Computes the standard contrastive divergence loss with entropy approximation based 
    on the paper: https://imrss2022.github.io/contributions/ta.pdf
    set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  state:      torch.Tensor of the training state batch
    :param  actions:   torch.Tensor of the ground truth action and the negative ones

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    true_action_energy = ebm(state[:, 0, :], actions)
    fake_action_energy = ebm(state[:, 1:, :], actions)

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = torch.pow(fake_action_energy, 2).mean()
    var_loss = torch.pow(fake_action_energy.mean(), 2)

    # add losses together
    loss = cd_loss + 0.5 * reg_loss - 0.5 * var_loss

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    # return the loss of the model for optimization
    return loss

###########################
# joint ebm loss functions to learn marginal EBMs


def compute_marginal_info_nce_loss(
    ebm: nn.Module, ebm_states: torch.Tensor, device: str, avrg_e_regularization: float
) -> torch.Tensor:
    """
    Computes the InfoNCE loss of the model based on the prediction of the
    ebm model and the ground truth action located at actions[:, 0, :].
    The InfoNCE loss can be understood as to classify the correct action from
    the set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  ebm_states: 
    :param  device:     torch.device

    :return:            torch.Tensor of the computed InfoNCE loss value
    """
    # add small noise to the positive sample
    ebm_states[:, 0, :] += torch.randn_like(ebm_states[:, 0, :]) * 1e-4
    
    # Generate a random permutation of the positives and negatives.
    permutation = torch.rand(ebm_states.size(0), ebm_states.size(1)).argsort(
        dim=1
    )  # [B, N+1]
    ebm_states = ebm_states[
        torch.arange(ebm_states.size(0)).unsqueeze(-1), permutation
    ]  # [B, N+1, D ]

    # Get the original index of the positive. This will serve as the class label
    # for the loss.
    ground_truth = (permutation == 0).nonzero()[:, 1].to(device)

    # For every element in the mini-batch, there is 1 positive for which the EBM
    # should output a low energy value, and N negatives for which the EBM should
    # output high energy values.
    energy = ebm(ebm_states)

    # get the average delta energy for debugging and training performance info
    delta_energy = energy[permutation==0].mean() - energy[permutation!=0].mean()
    # Interpreting the energy as a negative logit, we can apply a cross entropy loss
    # to train the EBM.
    logits = -1.0 * energy
    ground_truth = torch.unsqueeze(ground_truth, 1)
    loss = F.cross_entropy(logits, ground_truth)

    # add regularization loss
    reg_loss = torch.pow(energy, 2).mean()

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": energy.mean().item(),
            "min energy": energy.min().item(),
            "max energy": energy.max().item(),
            "delta energy": delta_energy,
            "avrg_positive_sample_energy": energy[permutation==0].mean(),
            "avrg_negative_sample_energy": energy[permutation!=0].mean(),
            "reg_loss": reg_loss,
            "loss_no_reg": loss
        }
    )
    loss = loss + avrg_e_regularization * reg_loss

    return loss


def marginal_contrastive_divergence(
    ebm: nn.Module, ebm_states: torch.Tensor, avrg_e_regularization: float
) -> torch.Tensor:
    """
    Computes the standard contrastive divergence loss for an ebm given a true action and a
    set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  ebm_states:      torch.Tensor of the training state batch

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    true_action_energy = ebm(ebm_states[:, 0, :])
    fake_action_energy = ebm(ebm_states[:, 1:, :])

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = (
        torch.pow(true_action_energy, 2).mean()
        + torch.pow(fake_action_energy, 2).mean()
    )

    # add losses together
    loss = cd_loss + avrg_e_regularization * reg_loss

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    # return the loss of the model for optimization
    return loss


def marginal_contrastive_divergence_kl(
    ebm: nn.Module, ebm_states: torch.Tensor, avrg_e_regularization: float, kl_loss_factor: float
) -> torch.Tensor:
    """
    Improved version of the contrastive divergence loss with additional KL based on the
    paper https://arxiv.org/pdf/2012.01316.pdf . The loss consists of the basic contrastive divergence
    term and in addition we approximate the entropy of the loss by using L2 distance of ....

    Code is based on the repo https://github.com/yilundu/improved_contrastive_divergence/blob/master/train.py
    to the paper mentioned above.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  ebm_states:      torch.Tensor of the training state batch

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    true_action_energy = ebm(ebm_states[:, 0, :])
    fake_action_energy = ebm(ebm_states[:, 1:, :])

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = (
        torch.pow(true_action_energy, 2).mean()
        + torch.pow(fake_action_energy, 2).mean()
    ).mean()

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    entropy_temperature = 1e-1
    # now compute the KL loss term
    # the KL loss term consists of two parts: the entropy loss part and the sampler energy loss part

    # first we need the gradient w.r.t to the mcmc sample:
    ebm.requires_grad_(False)
    ebm_states_2 = ebm_states.clone()
    ebm_states_2.requires_grad = True
    # compute the energy w.r.t to the state sample
    fake_action_energy_2 = ebm(ebm_states_2)

    # set the gradient of the ebm back to previous version
    ebm.requires_grad_(True)

    # compute the kl sampler loss part
    kl_opt_loss = fake_action_energy_2.mean()

    # we approximate the entropy loss term using nearest neighbor of our generated samples
    # therefore  we compute the distance of our generated samples
    nn_dist = torch.norm(
        fake_action_energy_2[:, None, Ellipsis]
        - fake_action_energy_2[:, :, None, Ellipsis],
        p=2,
        dim=-1,
    )
    # we sum it up to estimate the entropy of our samples
    # this loss part forces the network to generate diverse samples with low energy
    kl_entropy_loss = torch.log(nn_dist.min(dim=1)[0]).mean()

    # the -0.3 is based on the same loss implementation of the paper
    # finally we add our loss parts together to the total loss and return it
    loss = cd_loss + kl_entropy_loss - kl_loss_factor * kl_opt_loss + avrg_e_regularization*reg_loss
    return loss

def marginal_contrastive_divergence_entropy_approx(
    ebm: nn.Module,ebm_states: torch.Tensor
) -> torch.Tensor:
    """
    Computes the standard contrastive divergence loss with entropy approximation based 
    on the paper: https://imrss2022.github.io/contributions/ta.pdf
    set of false actions.

    :param  ebm:        nn.Module of the ebm to compute the loss for
    :param  ebm_states:      torch.Tensor of the training state batch

    :return:            torch.Tensor of the computed CD loss value
    """
    # get the energy of the true action and all sampled ones
    true_action_energy = ebm(ebm_states[:, 0, :])
    fake_action_energy = ebm(ebm_states[:, 1:, :])

    # compute the average difference in energy prediction for every action in the batch
    cd_loss = true_action_energy.mean(dim=1) - fake_action_energy.mean(dim=1)
    cd_loss = cd_loss.mean()
    # add regularization loss
    reg_loss = torch.pow(fake_action_energy, 2).mean()
    var_loss = torch.pow(fake_action_energy.mean(), 2)

    # add losses together
    loss = cd_loss + 0.5 * reg_loss - 0.5 * var_loss

    # log the average minimum and maximum energy of the current batch for additional debugging and information
    wandb.log(
        {
            "mean energy": true_action_energy.mean().item() / 2
            + fake_action_energy.mean().item() / 2,
            "min energy": true_action_energy.min().item(),
            "max energy": fake_action_energy.max().item(),
            "delta energy": cd_loss,
            "avrg_positive_sample_energy": true_action_energy.mean().item(),
            "avrg_negative_sample_energy": fake_action_energy.mean().item(),
        }
    )
    # return the loss of the model for optimization
    return loss