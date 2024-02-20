from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from torch.distributions import Categorical
from collections import deque
from typing import Optional, Tuple

import logging
import math 
from typing import Optional

import torch
import hydra
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops
from omegaconf import ListConfig, OmegaConf
import torch.nn as nn
import torch.nn.functional as F
import einops

from beso.networks.scaler.scaler_class import Scaler
from beso.networks.transformers.mingpt_policy import Block
from beso.agents.lmp_agents.lmp_modules.action_decoders.action_decoder import ActionDecoder
logger = logging.getLogger(__name__)


class GPTActionDecoder(ActionDecoder):

    def __init__(
            self,
            perceptual_features: int,
            latent_goal_features: int,
            plan_features: int,
            model: OmegaConf,
            device: str
        ) -> None:
        super(GPTActionDecoder, self).__init__()
        self.device = device
        self.scaler = None
        self.hidden_state = None
        self.perceptual_features = perceptual_features
        self.latent_goal_features = latent_goal_features
        self.plan_features = plan_features
        self.model = hydra.utils.instantiate(model).to(self.device)

        self.obs_context = deque(maxlen=model.obs_seq_len)
        self.goal_context = deque(maxlen=model.goal_seq_len)

    def set_bounds(self, scaler: Scaler, window_size, device):
        self.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(device)
        self.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(device)
    
    def loss_and_act(
        self, 
        latent_plan: torch.Tensor, 
        perceptual_emb: torch.Tensor, 
        latent_goal: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_pred = self.model(perceptual_emb, latent_goal, latent_plan)
        # loss
        loss = nn.MSELoss()(action_pred, actions).mean()
        return loss, action_pred

    def clear_hidden_state(self) -> None:
        self.obs_context.clear()
        self.goal_context.clear()

    def act(
        self, 
        latent_plan: torch.Tensor, 
        perceptual_emb: torch.Tensor, 
        latent_goal: torch.Tensor
    ) -> torch.Tensor:
        self.obs_context.append(perceptual_emb)
        self.goal_context.append(latent_goal)

        input_state = torch.concat(tuple(self.obs_context), dim=1)
        input_goal = torch.concat(tuple(self.goal_context), dim=1)
        action_pred = self.model(input_state, input_goal, latent_plan)
        if action_pred.size()[1] > 1 and perceptual_emb.shape[0] == 1:
            action_pred = action_pred[:, -1, :]
        return action_pred

    def loss(
        self, 
        latent_plan: torch.Tensor, 
        perceptual_emb: torch.Tensor, 
        latent_goal: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        pred_actions = self(latent_plan, perceptual_emb, latent_goal)
        return torch.function.mse_loss(actions, pred_actions)

    def forward(  # type: ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        if len(latent_plan.shape) == 2:
            latent_plan = einops.repeat(latent_plan, 'b d -> b t d', t=seq_len)
        else:
            latent_plan = einops.repeat(latent_plan, 'b 1 d -> b (t 1) d', t=seq_len) if latent_plan.shape[1] == 1 else latent_plan 
        if len(latent_plan.shape) == 2:
            latent_goal = einops.repeat(latent_goal, 'b d -> b t d', t=seq_len)  if latent_goal.shape[1] == 1 else latent_goal 
        else:
            latent_goal = einops.repeat(latent_goal, 'b 1 d -> b (t 1) d', t=seq_len) 
        x, h_n = self.model(latent_plan, perceptual_emb, latent_goal)
        return x, None


# simple BC GPT model variant, that predicts action directly
class LatentPlansBCGPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        state_dim: int,
        device: str,
        goal_conditioned: bool,
        action_dim: int,
        embed_dim: int,
        latent_plan_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        sigma_vocab_size: int,
        goal_drop: float = 0.1,
        linear_output = False,
        use_goal: bool = True,
    ):
        super(LatentPlansBCGPT, self).__init__()
        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + 2 * obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len + 1
        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.latent_plans_emb = nn.Linear(latent_plan_dim, embed_dim)
        self.drop = nn.Dropout(embed_pdrob)
        
        # needed for goal conditioning
        self.cond_mask_prob = goal_drop
        self.use_goal = use_goal
        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim
        # transformer
        self.blocks = nn.Sequential(
            *[Block(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size
            ) for _ in range(n_layers)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        # action pred module 
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), 
                nn.GELU(),  
                nn.Linear(100, self.action_dim)
            )
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, LatentPlansBCGPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(
        self, 
        states,
        goals,
        latent_plans,
        uncond: Optional[bool] =False,
    ):  
        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if len(latent_plans.shape) == 2:
            latent_plans = latent_plans.unsqueeze(1)
        # define the total length of the input sequence
        seq_length = self.goal_seq_len + 1 + t*2
        # get the beginning of the state action pairs
        
        if self.goal_conditioned and self.use_goal:
            second_half_idx = self.goal_seq_len + 1 
        else:
            second_half_idx = 1
        # during training randomly mask out the goal
        # to train the conditional model with classifier-free guidance wen need 
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2 
        if self.training:
            goals = self.mask_cond(goals)
            # latent_plans = self.mask_cond(latent_plans)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)  
            # latent_plans = torch.zeros_like(latent_plans).to(self.device)
        
        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        goal_embed = self.tok_emb(goals)
        latent_plan_embed = self.latent_plans_emb(latent_plans)
        
        # if not uncond:
        if self.goal_conditioned:
            position_embeddings = self.pos_emb[
            :, :(t + self.goal_seq_len), :
            ]  # each position maps to a (learnable) vector
        else: # without goal conditioning we only have the obs sequence 
            position_embeddings = self.pos_emb[
            :, :t, :
            ]
        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the masking would not make sense
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        
        # next we stack everything together 
        if self.goal_conditioned:
            if self.use_goal:
                input_seq = torch.cat([latent_plan_embed, goal_x, state_x], dim=1)
            else:
                input_seq = torch.cat([latent_plan_embed, state_x], dim=1)
        else:
            input_seq = state_x
        
        # Note we need to also adept the action masks 
        x = self.blocks(input_seq)
        x = self.ln_f(x)
        
        # now we want the last half of the output
        x = x[:, second_half_idx:, :]
        pred_actions = self.action_pred(x)
        return pred_actions
    
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob) 
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()