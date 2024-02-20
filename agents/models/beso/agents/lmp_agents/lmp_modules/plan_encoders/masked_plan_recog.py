import logging
import math 
from typing import Optional

from omegaconf import DictConfig
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hydra

from beso.networks.transformers.mingpt_policy import Block
from beso.agents.lmp_agents.lmp_modules.utils import PositionalEncoding
logger = logging.getLogger(__name__)


# inspired by https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py#L49
class MaskedPlanEncoder(nn.Module):
    '''
    Plan encoder for the masked LMP model. Inspired by the Masked Autoencoder for Images and Sequences (MAE) paper.
    '''
    def __init__(
        self,
        latent_dim: int,
        decoder_latent_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        masking_rate: float,
        window_size: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        encoder_normalize: bool,
        reconstruct_actions: bool,
        reconstruct_states: bool,
        num_heads: int,
        enc_dropout: float,
        device: str,
        dec_attn_pdrop: float,
        dec_resid_pdrop: float,
        use_goal_in_recognition: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.decoder_latent_dim = decoder_latent_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.masking_rate = masking_rate
        self.drop_last = drop_last
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.reconstruct_actions = reconstruct_actions
        self.reconstruct_states = reconstruct_states
        self.use_goal_in_recognition = use_goal_in_recognition
        self.window_size = window_size + 1
        # if we use the goal in the recognition, we need to add one more step
        if self.use_goal_in_recognition:
            self.window_size += 1
        self.enc_dropout = enc_dropout
        # get the linear encoders
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.obs_emb = nn.Linear(self.state_dim, self.latent_dim)
        self.action_emb = nn.Linear(self.action_dim, self.latent_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size+2, self.latent_dim))
        # self.pos_emb = PositionalEncoding(self.latent_dim, max_len=self.window_size)
        plan_encoder_layer = nn.TransformerEncoderLayer(
            self.latent_dim,
            self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.enc_dropout,
            batch_first=True
        )
        self.encoder_norm = nn.LayerNorm(self.latent_dim ) if encoder_normalize else None
        self.decoder_norm = nn.LayerNorm(self.decoder_latent_dim) 
        # self.layernorm = nn.LayerNorm(self.latent_dim )
        self.enc_dropout = nn.Dropout(p=enc_dropout)
        self.transformer_encoder = nn.TransformerEncoder(plan_encoder_layer, num_layers=num_encoder_layers, norm=self.encoder_norm)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(self.latent_dim, self.decoder_latent_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.window_size, self.decoder_latent_dim), requires_grad=False)  # fixed sin-cos embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_latent_dim))
        
        plan_decoder_layer = nn.TransformerEncoderLayer(
            self.decoder_latent_dim,
            self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=dec_attn_pdrop,
            batch_first=True
        )
        # self.blocks = nn.TransformerEncoder(plan_decoder_layer, num_layers=num_decoder_layers, norm=self.decoder_norm)
        self.blocks = nn.Sequential(
            *[Block(
                self.decoder_latent_dim,
                self.num_heads,
                dec_attn_pdrop,
                dec_resid_pdrop,
                self.window_size,
            ) for _ in range(self.num_decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_latent_dim)
        # decoder head
        self.ln_f = nn.LayerNorm(self.decoder_latent_dim)
        if not self.reconstruct_actions and not self.reconstruct_states:
            self.reconstruct_states = True
            # raise ValueError("At least one of reconstruct_actions or reconstruct_states must be True")
        if self.reconstruct_actions:
            self.action_head = nn.Linear(self.decoder_latent_dim, self.action_dim)
        if self.reconstruct_states:
            self.state_head = nn.Linear(self.decoder_latent_dim, self.state_dim)
        self.proj_model = nn.Sequential(
            nn.Linear(self.decoder_latent_dim*2, self.decoder_latent_dim*2),
            nn.Mish(),
            nn.Linear(self.decoder_latent_dim*2, self.decoder_latent_dim),
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, MaskedPlanEncoder):
            if isinstance(self.pos_emb, PositionalEncoding):
                pass
            else:
                torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # torch.nn.init.normal_(self.proj_model, std=.02)
             
    def forward(self, input_seq, actions):
        # encoder
        latent_x, mask, ids_restore = self.forward_encoder(input_seq)
        # decoder
        pred_recon = self.forward_decoder(latent_x, ids_restore)
        action_loss, states_loss = self.forward_loss(actions, input_seq, pred_recon, mask)
        # the first output is the latent representation of the input sequence using the CLS token
        sampled_plan = self.proposal_forward(input_seq[:, 0, :], input_seq[:, -1, :])
        proposal_loss = F.l1_loss(sampled_plan, latent_x[:, 0, :].detach())
        return sampled_plan, action_loss, states_loss, proposal_loss
           
    def forward_encoder(self, x):
        # embed input states
        x = self.obs_emb(x)
        # add pos embed w/o cls token
        if isinstance(self.pos_emb, PositionalEncoding):
            pass
        else:
            x = x + self.pos_emb[:, 1:, :]
        # first_state = x[:, 0, :]
        if self.drop_last:
            last_state = x[:, -1, :].unsqueeze(1)
            middle_states = x[:, :-1, :]
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(middle_states, self.masking_rate)
        else:
            x, mask, ids_restore = self.random_masking(x, self.masking_rate)

        # append cls token
        if isinstance(self.pos_emb, PositionalEncoding):
            pass
        else:
            cls_token = self.cls_token + self.pos_emb[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        if self.drop_last:
            x = torch.cat((cls_tokens, x, last_state), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        if isinstance(self.pos_emb, PositionalEncoding):
            x = self.pos_emb(x)
        # apply Transformer blocks
        x = self.transformer_encoder(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore
    
    def forward_loss(self, target_actions, target_states, pred, mask):
        """
        TODO add docstring
        """
        if self.reconstruct_actions and self.reconstruct_states:
            # remove the cls token
            pred_states = pred[1] # [:, 1:, :]
            pred_actions = pred[0] # [:, 1:, :]
            # mask = mask[:, 1:]
            # if we use the goal in recognition, we need to remove the last action 
            # since there is no target action for the future goal
            if self.use_goal_in_recognition:
                target_actions = target_actions[:, :, :]
                pred_actions = pred_actions[:, :-1, :]
                # target_states = target_states[:, :-1, :]
            action_loss = (pred_actions - target_actions) ** 2
            action_loss = action_loss.mean(dim=-1).mean()  # [N, L], mean loss per patch
            state_loss = (pred_states - target_states) ** 2
            state_loss = state_loss.mean(dim=-1).mean()  # [N, L], mean loss per patch
            
            return action_loss, state_loss

        elif self.reconstruct_actions and not self.reconstruct_states:
            # remove the cls token
            # pred = pred[:, 1:, :]
            # mask = mask[:, 1:]
            # if we use the goal in recognition, we need to remove the last action 
            # since there is no target action for the future goal
            if self.use_goal_in_recognition:
                pred = pred[:, :-1, :]
                # mask = mask[:, :-1]
            loss = (pred - target_actions) ** 2
            loss = loss.mean(dim=-1).mean()  # [N, L], mean loss per patch
            # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            return loss, None
        
        elif self.reconstruct_states and not self.reconstruct_actions:
            # remove the cls token
            # pred = pred[:, 1:, :]
            # mask = mask[:, 1:]
             # if we use the goal in recognition, we need to remove the last action 
            # since there is no target action for the future goal
            if self.use_goal_in_recognition:
                pred = pred[:, :-1, :]
                mask = mask[:, :-1]
                target_states = target_states[:, :-1, :]
            loss = (pred - target_states) ** 2
            loss = loss.mean(dim=-1).mean()
            return None, loss

    def forward_decoder(self, latent, ids_restore):
        # decoder embed tokens
        x = self.decoder_embed(latent)
        if self.drop_last:
            x = x[:, :-1, :]
            last_state = x[:, -1, :].unsqueeze(1)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # append cls token
        if self.drop_last:
            x = torch.cat([x[:, :1, :], x_, last_state], dim=1)
        else:
            x = torch.cat([x[:, :1, :], x_], dim=1)  
        
        # add pos embed
        x = x + self.decoder_pos_embed #[:, :, :]
        
        # decoder blocks
        pred = self.blocks(x)
        pred = self.decoder_norm(pred)
        # remove cls token for state and action prediction
        pred = pred[:, 1:, :]
        # pred actions and/or states
        if self.reconstruct_actions and not self.reconstruct_states:
            action_pred = self.action_head(pred)
            return action_pred
        elif self.reconstruct_states and not self.reconstruct_actions:
            state_pred = self.state_head(pred)
            return state_pred
        elif self.reconstruct_states and self.reconstruct_actions:
            action_pred = self.action_head(pred)
            state_pred = self.state_head(pred)
            return action_pred, state_pred
    
    @torch.no_grad()
    def predict(self, input_seq):
        latent_plan = self.get_latent_plan(input_seq)
        return latent_plan
    
    def proposal_forward(self, first_state, goal_state):
        if len(first_state.shape) == 2:
            first_state = first_state.unsqueeze(1)
        if len(goal_state.shape) == 2:
            goal_state = goal_state.unsqueeze(1)
        input_seq = torch.cat([first_state, goal_state], dim=1)
        x =  self.obs_emb(input_seq)
        x = x + torch.cat([self.pos_emb[:, 1, :].unsqueeze(1), self.pos_emb[:, -1, :].unsqueeze(1)  ], dim=1)
        cls_token = self.cls_token + self.pos_emb[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_encoder(x)
        x = self.encoder_norm(x)
        return x[:, 0, :]
    
    def get_latent_plan(self, x):
        # embed input states
        x = self.obs_emb(x)
        # add pos embed w/o cls token
        if isinstance(self.pos_emb, PositionalEncoding):
            pass
        else:
            x = x + self.pos_emb[:, 1:, :]
        
        # append cls token
        if isinstance(self.pos_emb, PositionalEncoding):
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + self.pos_emb[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        if isinstance(self.pos_emb, PositionalEncoding):
            x = self.pos_emb(x)
        # apply Transformer blocks
        x = self.transformer_encoder(x)
        x = self.encoder_norm(x)
        # only return the latent plan from the cls token
        return x[:, 0, :]
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

