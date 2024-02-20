
import torch 
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import hydra

from beso.networks.transformers.mingpt_policy import Block


class VqVaeGPT(nn.Module):
    '''
    Transformer model for discrete latent space plan prior, which is used to generate the next plan for VQ-VAE Planners.
    '''
    def __init__(
        self,
        state_dim,
        pkeep,
        device,
        latent_dim,
        num_layers,
        num_heads,
        window_size,
        dec_attn_pdrop,
        dec_resid_pdrop,
        vocab_size,
        embd_pdrop,
        block_size,
        top_k,
        ) -> None:
        super().__init__()
        self.device = device 
        self.pkeep = pkeep
        self.top_k = top_k
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.gpt = PlanGPT(
            latent_dim,
            state_dim,
            num_layers,
            num_heads,
            window_size,
            dec_attn_pdrop,
            dec_resid_pdrop,
            vocab_size,
            embd_pdrop,
            block_size,
        )
    
    def forward(self, first_state, goal_state, indices):
        
        target = indices
        logits, _= self.gpt(goal_state, first_state)
        logits = logits[:, -1:, :]
        return logits, target
    
    def forward_masking(self, first_state, goal_state, indices):
        '''
        Masking the input sequence with probability pkeep.
        '''
        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=self.device))
        mask = mask.round().to(dtype=torch.int64)
        
        random_indices = torch.randint(0, self.vocab_size, indices.shape, device=self.device)
        
        new_indices = indices * mask + random_indices * (1 - mask)
        
        target = indices
        
        logits, _= self.gpt(goal_state, first_state, new_indices[:, :-1])
        
        return logits, target
    
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k=k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out 
        
        
    @torch.no_grad()
    def sample(self, first_state, goal, steps, temperature=1.0):
        '''
        Method to sample class labels from the model as the next plan.
        '''
        self.gpt.eval()
        context_length = first_state.shape[1] + goal.shape[1]
        for k in range(steps):
            if k == 0:
                logits, _ = self.gpt(goal, first_state)
            
            else:
                logits, _ = self.gpt(goal, first_state, x)
            logits = logits[:, -1, :] / temperature
            
            if self.top_k is not None: 
                logits = self.top_k_logits(logits, self.top_k)
                
            probs = F.softmax(logits, dim=-1)
            
            ix = torch.multinomial(probs, num_samples=1)
            if k == 0:
                x = ix
            else:
                x = torch.cat((x, ix), dim=1)
        
        # onlt get the plan and not the context
        x = x[:, (context_length - 2):]
        return x
        


class PlanGPT(nn.Module):
    '''
    Minining GPT model for discrete latent space plan prior.
    '''
    def __init__(
        self,
        latent_dim,
        state_dim,
        num_layers,
        num_heads,
        window_size,
        dec_attn_pdrop,
        dec_resid_pdrop,
        vocab_size,
        embd_pdrop,
        block_size,
        ) -> None:
        super().__init__()
        
        self.blocks = nn.Sequential(
            *[Block(
                latent_dim,
                num_heads,
                dec_attn_pdrop,
                dec_resid_pdrop,
                window_size,
            ) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(latent_dim)
        self.state_embd = nn.Linear(state_dim, latent_dim)
        self.head = nn.Linear(latent_dim, vocab_size)
        
        self.tok_emb = nn.Embedding(vocab_size, latent_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, latent_dim))  # 512 x 1024
        self.drop = nn.Dropout(embd_pdrop)

        self.block_size = block_size
        self.apply(self._init_weights)
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, goal, state, idx=None):
        '''
        Forward pass of the model to generate the next token of the plan.
        '''
        embd_state = self.state_embd(state)
        embd_goal = self.state_embd(goal)
        context_sequence = torch.cat((embd_state, embd_goal), dim=1)
        
        if idx is not None:
            token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector

            t = token_embeddings.shape[1] + context_sequence.shape[1]
            input_sequence = torch.cat((context_sequence, token_embeddings), dim=1)
        else:
            t = context_sequence.shape[1]
            input_sequence = context_sequence

        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(input_sequence + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits, None
        
        