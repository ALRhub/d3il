"""
Transformer CVAE model for ACT

Model from with additional Hydra configs for easier experimentation

https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py

"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import hydra
from omegaconf import DictConfig
from torch.nn import functional as F
import math
import logging

import IPython

e = IPython.embed

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
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
        k = (self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
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


class CausalSelfCrossAttention(nn.Module):
    def __init__(self, n_embd, cross_embed, n_heads, attn_pdrop, resid_pdrop, block_size):
        super().__init__()

        assert n_embd % n_heads == 0

        # Self-Attention Projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Cross-Attention Projections
        self.cross_key = nn.Linear(cross_embed, n_embd)
        self.cross_query = nn.Linear(n_embd, n_embd)
        self.cross_value = nn.Linear(cross_embed, n_embd)

        # Regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Output Projection
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask for Self-Attention
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.n_head = n_heads

    def forward(self, x, cross_input=None):
        B, T, C = x.size()

        # calculate query, key, values for self-attention
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v

        if cross_input is not None:
            # calculate query, key, values for cross-attention
            T_C = cross_input.size(1)
            k_cross = self.cross_key(cross_input).view(B, T_C, self.n_head, C // self.n_head).transpose(1, 2)
            v_cross = self.cross_value(cross_input).view(B, T_C, self.n_head, C // self.n_head).transpose(1, 2)

            q_cross = self.cross_query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # cross-attention
            att_cross = (q_cross @ k_cross.transpose(-2, -1)) * (1.0 / math.sqrt(k_cross.size(-1)))
            att_cross = F.softmax(att_cross, dim=-1)
            att_cross = self.attn_drop(att_cross)
            y_cross = att_cross @ v_cross

            # combine self-attention and cross-attention
            y = y + y_cross  # or any other combination strategy

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        return y


class DecoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
            self,
            n_embd: int,
            cross_embd: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            block_size: int,

    ):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, bias=False)
        self.ln2 = LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfCrossAttention(
            n_embd,
            cross_embd,
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

    def forward(self, x, cond=None):
        x = x + self.attn(self.ln1(x), cross_input=cond)
        x = x + self.mlp(self.ln2(x))
        return x


class EncoderBlock(nn.Module):
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
        self.ln1 = LayerNorm(n_embd, bias=False)
        self.ln2 = LayerNorm(n_embd, bias=False)
        self.attn = SelfAttention(
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


class TransformerEncoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int,
            bias: bool = False,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[EncoderBlock(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size,
            )
                for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.ln(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            cross_embed: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int,
            bias: bool = False,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[DecoderBlock(
                embed_dim,
                cross_embed,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size,
            )
                for _ in range(n_layers)]
        )
        self.ln = LayerNorm(embed_dim, bias)

    def forward(self, x, cond=None):
        for layer in self.blocks:
            x = layer(x, cond=cond)
        x = self.ln(x)
        return x


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ActVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(
            self,
            action_encoder: DictConfig,
            encoder: DictConfig,
            decoder: DictConfig,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            act_seq_size: int,
            latent_dim: int = 32,
            goal_dim=None
    ):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: size of the action sequence
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder)
        self.decoder = hydra.utils.instantiate(decoder)
        # general linear projection for the encoder
        self.state_encoder = nn.Linear(state_dim, hidden_dim, bias=False)
        if goal_dim is not None:
            self.goal_encoder = nn.Linear(goal_dim, hidden_dim, bias=False)
        self.action_embed = nn.Linear(action_dim, action_encoder.embed_dim, bias=False)
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim, bias=False)

        # the output heads for the decoder and action encoder
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.latent_proj = nn.Linear(action_encoder.embed_dim, latent_dim * 2)
        # get the tokens for the action decoder
        self.query_embed = nn.Embedding(act_seq_size, hidden_dim)

        # we need a second transformer encoder for the action decoder
        self.action_encoder = hydra.utils.instantiate(action_encoder)

        # encoder extra parameters
        self.latent_dim = latent_dim  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, action_encoder.embed_dim)  # extra cls token embedding

        self.pos_emb = nn.Parameter(torch.zeros(1, act_seq_size, hidden_dim))
        self.act_pos_emb = nn.Parameter(torch.zeros(1, act_seq_size + 1, action_encoder.embed_dim))
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ActVAE):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, state, goal=None, action=None):

        state = state[:, :1, :]

        bs = state.shape[0]

        state_embed = self.state_encoder(state)
        if goal is not None:
            goal_embed = self.goal_encoder(goal)

        # if the action is not None, we are in the training phase
        # thus we need to encode the action sequence into the latent space
        if action is not None:
            action_embed = self.action_embed(action)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            action_seq = torch.cat([cls_embed, action_embed], dim=1)

            action_seq = action_seq + self.act_pos_emb

            # we get the latent z from the first cls token
            latent_z = self.action_encoder(action_seq)[:, 0].unsqueeze(1)
            # next we get the mean and logvar of the latent z
            latent_info = self.latent_proj(latent_z)
            mu = latent_info[:, :, :self.latent_dim]
            logvar = latent_info[:, :, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        # if the action is None, we are in the testing phase and sample from the prior
        else:
            mu = logvar = None
            latent_sample = torch.rand([bs, 1, self.latent_dim], dtype=torch.float32).to(state.device)
            # latent_sample = torch.zeros([bs, 1, self.latent_dim], dtype=torch.float32).to(state.device)
            latent_input = self.latent_out_proj(latent_sample)

        # next we concatenate the latent z with the state and goal for the encoder

        if goal is not None:
            encoder_input = torch.cat([state_embed, goal_embed, latent_input], dim=1)
        else:
            encoder_input = torch.cat([state_embed, latent_input], dim=1)

        # add the positional embedding
        encoder_input = encoder_input + self.pos_emb[:, :encoder_input.shape[1]]

        # encode the state, goal and latent z into the hidden dim
        encoder_output = self.encoder(encoder_input)

        # decode the action sequence with cross attention over the encoder output
        action_seq = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        decoder_output = self.decoder(action_seq, encoder_output)

        # get the action prediction from the head
        action_pred = self.action_head(decoder_output)

        return action_pred, [mu, logvar]

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

    def get_params(self):
        return self.parameters()