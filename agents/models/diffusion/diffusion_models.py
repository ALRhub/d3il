import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import einops
import math
from typing import Optional
from torch.nn import functional as F
import logging

from agents.models.common.utils import return_activiation_fcn
from agents.models.common.utils import CrossAttention, LayerNorm
from agents.models.common.mlp import ResidualMLPNetwork, MLPNetwork

from .utils import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class DiffusionMLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network for benchmarking the performance of different networks. The model is used in
    several papers and can be used to compare all_plots model performances.
    """

    def __init__(
            self,
            action_dim: int,
            obs_dim: int,
            t_dim: int,
            residual_style: bool = True,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            use_spectral_norm: bool = False,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            device: str = 'cuda',
            goal_conditioned: bool = True,
            cond_mask_prob: float = 0
    ):
        super(DiffusionMLPNetwork, self).__init__()
        self.network_type = "mlp"
        self.device = device
        self.goal_conditioned = goal_conditioned
        self.cond_mask_prob = cond_mask_prob
        # number of samples per batch
        output_dim = action_dim
        # define the input dimension for the model
        if self.goal_conditioned:
            input_dim = 2 * obs_dim + action_dim + t_dim
        else:
            input_dim = obs_dim + action_dim + t_dim
        # time embedding model
        self.temp_layers = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.temp_layers.to(self.device)
        # set up the network
        if residual_style:
            self.layers = ResidualMLPNetwork(
                input_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                use_norm,
                norm_style,
                device
            ).to(device)
        else:
            self.layers = MLPNetwork(
                input_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                device
            ).to(device)
        self.layers.to(self.device)

    def forward(self, x, t, state, goal):
        # during training randomly mask out the goal
        t = self.temp_layers(t)

        if len(state.shape) == 3:
            # x = einops.rearrange(x, 'batch dim-> batch 1 dim')
            t = einops.rearrange(t, 'batch dim-> batch 1 dim')
            if self.goal_conditioned:
                # check cond mask prob
                if self.training:
                    goal = self.mask_cond(goal, self.mask_cond)
                x = torch.cat([x, t, state, goal], dim=2)
            else:
                x = torch.cat([x, t, state], dim=2)
        else:
            if self.goal_conditioned:
                # check cond mask prob
                if self.training:
                    goal = self.mask_cond(goal, self.mask_cond)
                x = torch.cat([x, t, state, goal], dim=1)
            else:
                x = torch.cat([x, t, state], dim=1)
        out = self.layers(x)

        return out

    def get_device(self, device: torch.device):
        self.device = device
        self.layers.to(device)

    def get_params(self):
        return self.parameters()

    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d),
                                              device=cond.device) * self.cond_mask_prob)  # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


class LinearCrossAttentionLayer(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            condition_dim,
            dim_head,
            device,
            activation: str = 'Mish'
    ) -> None:
        super(LinearCrossAttentionLayer, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.dim_head = dim_head
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.input_dim, self.hidden_dim),
                CrossAttention(
                    query_dim=self.hidden_dim,
                    context_dim=self.condition_dim,
                    dim_head=self.dim_head
                )

            ]
        )
        self.norm1 = LayerNorm(self.input_dim)
        self.norm2 = LayerNorm(self.hidden_dim)
        self.act = return_activiation_fcn(activation)

    def forward(self, x, cond):
        x = self.layers[0](self.norm1(x))
        y = self.layers[1](self.norm2(x), context=cond, mask=None) + x
        return y


class LatentDiffusionMLPNetwork(nn.Module):

    def __init__(
            self,
            action_dim: int,
            obs_dim: int,
            t_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            dropout: int = 0,
            dim_head=16,
            activation: str = "ReLU",
            device: str = 'cuda'
    ):
        self.network_type = "mlp"
        self.device = device
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.t_dim = t_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        # number of samples per batch
        self.output_dim = action_dim
        self.dropout = dropout
        self.dim_head = dim_head
        self.temp_layers = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.temp_layers.to(self.device)
        self.act = return_activiation_fcn(activation)


class GCDiffusionAttentionMLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network for benchmarking the performance of different networks. The model is used in
    several papers and can be used to compare all_plots model performances.
    """

    def __init__(
            self,
            action_dim: int,
            obs_dim: int,
            t_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            dropout: int = 0,
            dim_head=16,
            activation: str = "ReLU",
            device: str = 'cuda'
    ):
        super(GCDiffusionAttentionMLPNetwork, self).__init__()
        self.network_type = "mlp"
        self.device = device
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.t_dim = t_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        # number of samples per batch
        self.output_dim = action_dim
        self.dropout = dropout
        self.dim_head = dim_head

        self.input_dim = self.action_dim + self.t_dim  # self.obs_dim +

        self.temp_layers = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.temp_layers.to(self.device)
        # set up the network
        self.layers = nn.ModuleList(
            [
                LinearCrossAttentionLayer(
                    self.input_dim,
                    self.hidden_dim,
                    condition_dim=self.obs_dim,
                    dim_head=self.dim_head,
                    device=device,
                    activation=activation
                )
            ]
        )
        # self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        # self.layers.extend([CrossAttention(query_dim=self.hidden_dim, context_dim=self.obs_dim, dim_head=self.dim_head)])
        for i in range(1, self.num_hidden_layers):
            self.layers.extend(
                [
                    LinearCrossAttentionLayer(
                        self.hidden_dim,
                        self.hidden_dim,
                        condition_dim=self.obs_dim,
                        dim_head=self.dim_head,
                        device=device,
                        activation=activation
                    )
                    # nn.Linear(self.hidden_dim, self.hidden_dim),
                    # CrossAttention(query_dim=self.hidden_dim, context_dim=self.obs_dim, dim_head=self.dim_head)
                ]
            )
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # build the activation layer
        self.act = return_activiation_fcn(activation)
        self.layers.to(self.device)

    def forward(self, x, t, state):

        t = self.temp_layers(t)
        x = torch.cat([x, t], dim=1)

        for idx, layer in enumerate(self.layers):
            if idx < len(self.layers) - 1:
                x = layer(x, state)
            else:
                out = layer(x)
        return out

    def get_device(self, device: torch.device):
        self.device = device
        self.layers.to(device)

    def get_params(self):
        return self.parameters()


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        ):
        super().__init__()
        assert n_embd % n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_heads

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
            self,
            n_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,

    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DiffusionTransformerNetwork(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            state_dim: int,
            device: str,
            goal_conditioned: bool,
            action_dim: int,
            embed_dim: int,
            embed_pdrob: float,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            n_heads: int,
            goal_seq_len: int,
            obs_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()
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
        self.tok_emb.to(self.device)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

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
        self.blocks.to(self.device)
        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.ln_f.to(self.device)

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

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

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
            keep_last_actions: Optional[bool] = False
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        if len(actions.size()) != 3:
            actions = actions.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = einops.rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:
            second_half_idx = self.goal_seq_len + 1
        else:
            second_half_idx = 1
        # during training randomly mask out the goal
        # to train the conditional model with classifier-free guidance wen need
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        # if not uncond:
        if self.goal_conditioned:
            position_embeddings = self.pos_emb[
                                  :, :(t + self.goal_seq_len), :
                                  ]  # each position maps to a (learnable) vector
        else:  # without goal conditioning we only have the obs sequence
            position_embeddings = self.pos_emb[
                                  :, :t, :
                                  ]
        # note, that the goal states are at the beginning of the sequence since they are available
        # for all states s_1, ..., s_t otherwise the masking would not make sense
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        # the action get the same position embedding as the related states
        action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])

        # now for the complicated part
        # we need to stack the input in the following order:
        # [sigma_emb, s_g1, ..., sg_n, s_1, a_1, s_2, a_2, ..., s_n, a_n]
        # first stack actions and states in the way: [s_1, a_1, s_2, a_2, ...,]
        sa_seq = torch.stack([state_x, action_x], dim=1
                             ).permute(0, 2, 1, 3).reshape(b, 2 * t, self.embed_dim)

        # next we stack everything together
        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, goal_x, sa_seq], dim=1)
        else:
            input_seq = torch.cat([emb_t, sa_seq], dim=1)

        # Note we need to also adept the action masks
        x = self.blocks(input_seq)
        x = self.ln_f(x)

        # now we want the last half of the output
        x = x[:, second_half_idx:, :]
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # we need to check this for inference and adapt the max seq len accord
        if x.size()[1] < 2 * self.obs_seq_len:
            x_len = int(x.size()[1] / 2)
            x = x.reshape(b, x_len, 2, self.embed_dim).permute(0, 2, 1, 3)
        else:
            x = x.reshape(b, self.obs_seq_len, 2, self.embed_dim).permute(0, 2, 1, 3)
        # get the outputs related to the actions
        action_outputs = x[:, 1]
        pred_actions = self.action_pred(action_outputs)
        if keep_last_actions:
            pred_actions = torch.cat([actions[:, :-1, :], pred_actions[:, -1, :].reshape(1, 1, -1)], dim=1)

        return pred_actions

    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # TODO Check which one is correct
            mask = torch.bernoulli(torch.ones((bs, t, d),
                                              device=cond.device) * self.cond_mask_prob)  # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            # mask = torch.bernoulli(torch.ones((bs, t, 1), device=cond.device) * self.cond_mask_prob)
            # mask = einops.repeat(mask, 'b t 1 -> b t (1 d)', d=d)
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()


class DiffusionEncDec(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        self.decoder = hydra.utils.instantiate(decoder)

        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # we need another embedding for the time
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.time_emb.to(self.device)
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_emb.to(self.device)
        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

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

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            actions,
            time,
            states,
            goals,
            uncond: Optional[bool] = False,
            keep_last_actions: Optional[bool] = False
    ):

        # actions = actions[:, self.obs_seq_len-1:, :]
        # states = states[:, :self.obs_seq_len, :]

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        if len(actions.size()) != 3:
            actions = actions.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the time embedding
        times = einops.rearrange(time, 'b -> b 1')
        emb_t = self.time_emb(times)

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]
        # note, that the goal states are at the beginning of the sequence since they are available
        # for all states s_1, ..., s_t otherwise the masking would not make sense
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])

        # state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        # # the action get the same position embedding as the related states
        # action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])

        input_seq = torch.cat([emb_t, state_x], dim=1)

        # encode the state, goal and latent z into the hidden dim
        encoder_output = self.encoder(input_seq)

        decoder_output = self.decoder(action_x, encoder_output)

        pred_actions = self.action_pred(decoder_output)

        return pred_actions

    def get_params(self):
        return self.parameters()
