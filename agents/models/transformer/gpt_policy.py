import torch
import torch.nn as nn

import agents.models.bet.libraries.mingpt.model as mingpt_model


# based on the code of BeT
class MinGPT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        block_size: int = 128,
        vocab_size: int = 50257,
        action_dim: int = 0,
        discrete_input: bool = False,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.action_dim = action_dim

        for k, v in kwargs.items():
            setattr(self, k, v)

        gpt_config = mingpt_model.GPTConfig(
            input_size=self.input_size,
            vocab_size=self.action_dim, #self.vocab_size,
            block_size=self.block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            discrete_input=discrete_input,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )

        self.model = mingpt_model.GPT(gpt_config)

    def get_params(self):
        '''
        Helper method to get all model parameters
        '''
        return self.model.parameters()

    def forward(self, obs_rep: torch.Tensor):

        output, _ = self.model(obs_rep)

        return output
