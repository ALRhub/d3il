#!/usr/bin/env python3

from typing import Tuple

import torch
from torch.distributions import Independent, Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import hydra
import einops 
from omegaconf import DictConfig
import numpy as np 

from beso.agents.lmp_agents.lmp_modules.utils import * 
from beso.networks.vaes.vq_vaes.v_quantizer_layer import VectorQuantizer, VectorQuantizerEMA
from beso.networks.transformers.mingpt_policy import Block
import torch.distributed as dist


class Codebook(nn.Module):
    '''
    Simple codebook module that maps each discrete token to a vector.
    '''
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            beta: float,
            reset_freq: int = 1000,
            reset_prob: float = 0.5,
        ) -> None:
        super().__init__()
        self.beta = beta
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        
        self.reset_freq = reset_freq
        self.reset_prob = reset_prob
        self.steps_since_reset = 0
        self.reset_counter = 0


    def forward(self, z):

        z = z.view(-1, self.embedding_dim)

        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t()) 
        
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        
        # encodings 
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(min_encoding_indices .shape[0], self.num_embeddings).to(z.device)
        encodings.scatter_(1, min_encoding_indices , 1)

        z_q = self.embedding(min_encoding_indices).squeeze(1)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Periodically update the codebook
        self.periodic_codebook_reset(z_q, indices=min_encoding_indices)

        return z_q, min_encoding_indices, loss, perplexity
    
    def periodic_codebook_reset(self, embeddings, indices):
        self.reset_counter += 1
        
        if self.reset_counter % self.reset_freq == 0:
            if np.random.rand() < self.reset_prob:
                # compute the new codebook values
                new_embedding = self.embedding.weight.clone()
                for i in range(self.embedding.weight.size(0)):
                    mask = indices == i
                    if mask.sum() > 0:
                        new_embedding[i, :] = embeddings[mask[:, 0], :].mean(dim=0)

                # update the codebook
                self.embedding.weight = nn.Parameter(new_embedding)
                print('Codebook updated!')
            
            
# adapted from https://github.com/swasun/VQ-VAE-Images/blob/master/src/vector_quantizer_ema.py
class EMACodebook(nn.Module):
    '''
    Codebook with exponential moving average updates of the codebook
    '''
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            beta: float,
            decay: float,
            device: str,
            epsilon=1e-5,
            reset_freq: int = 500,
            reset_prob: float = 0.6,
        ) -> None:
        super().__init__()
        self.beta = beta
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        self.device = device
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        # get an ema variant 
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon
        
        self.global_step = 0 
        self.reset_freq = reset_freq
        self.reset_prob = reset_prob
        self.steps_since_reset = 0
        self.reset_counter = 0

    def forward(self, z):

        z = z.view(-1, self.embedding_dim)

        # Calculate distances
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t()) 
        
        # encodings 
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(min_encoding_indices .shape[0], self.num_embeddings).to(self.device)
        encodings.scatter_(1, min_encoding_indices , 1)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)
    
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.num_embeddings * self._epsilon) * n
            )
            
            dw = torch.matmul(encodings.t(), z)
            self.ema_w = nn.Parameter(self.ema_w * self._decay + (1 - self._decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self._ema_cluster_size.unsqueeze(1))
            
        z_q = self.embedding(min_encoding_indices).squeeze(1)

        # loss 
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # compute quantized vectors
        z_q = z + (z_q - z).detach()

        # Periodically update the codebook
        if self.training:
            self.periodic_codebook_reset(z_q, indices=min_encoding_indices)
        # return the values and loss, perplexity monitors the quality of the codebook
        return z_q, min_encoding_indices, loss, perplexity

    def periodic_codebook_reset(self, embeddings, indices):
        '''
        In the original paper, the authors periodically update the codebook by averaging the embeddings
        from the Jukebox paper, they use a random number to decide whether to update the codebook or not
        '''
        self.reset_counter += 1
        
        if self.reset_counter % self.reset_freq == 0:
            if np.random.rand() < self.reset_prob:
                # compute the new codebook values
                new_embedding = self.embedding.weight.clone()
                for i in range(self.embedding.weight.size(0)):
                    mask = indices == i
                    if mask.sum() > 0:
                        new_embedding[i, :] = embeddings[mask[:, 0], :].mean(dim=0)

                # update the codebook
                self.embedding.weight = nn.Parameter(new_embedding)
                print('Codebook updated!')
        
class VQGAN(nn.Module):

    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            num_embeddings: int,
            beta: float,
            embedding_dim: int,
            disc_factor: float,
            treshold: float,
            use_discriminator: bool,
            discriminator: DictConfig,
            lr: float,
            eps: float,
            betas: Tuple[float, float],
        ) -> None:
        super().__init__()
        self.plan_features = embedding_dim
        self.use_discriminator = use_discriminator
        self.encoder = hydra.utils.instantiate(encoder)
        self.decoder = hydra.utils.instantiate(decoder)
        self.codebook = Codebook(num_embeddings, embedding_dim, beta)
        self.discriminator = hydra.utils.instantiate(discriminator)
        
        self.disc_factor = disc_factor
        self.treshold = treshold
        
        self.opt_vq, self.opt_disc = self.configure_optimizers(lr, eps, betas)
        
    
    def configure_optimizers(self, lr, eps, betas):
        opt_vq = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.codebook.parameters()), 
            lr=lr, eps=eps, betas=betas)
        if self.use_discriminator:
            opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, eps=eps, betas=betas)
        else:
            None
        return opt_vq, opt_disc
        

    def forward(self, x):
        encoded_states = self.encoder(x)
        quant_states, codebook_indices, q_loss = self.codebook(encoded_states)
        decoded_states = self.decoder(quant_states)
        return decoded_states, codebook_indices, q_loss
    
    def compute_loss_and_plan(self, states, actions):
        
        pred_states, codebook_indices, q_loss = self.forward(states)
        
        if self.use_discriminator:
            disc_real = self.discriminator(states)
            disc_fake = self.discriminator(pred_states)
        
        vq_loss, gan_loss = self.compute_loss(states, pred_states, disc_real, disc_fake, q_loss)
        
        self.opt_vq.zero_grad()
        vq_loss.backward()
        
        if self.use_discriminator:
            self.opt_disc.zero_grad()
            gan_loss.backward()
        
        self.opt_vq.step()
        if self.use_discriminator:
            self.opt_disc.step()
        
        if self.use_discriminator:
            return vq_loss.item(), gan_loss.item()
        else:
            return vq_loss.item(), 0
        
    def compute_loss(self, states, pred_states, disc_real, disc_fake, q_loss):
        
        # reconstruction stuff 
        reconstruction_loss = F.mse_loss(pred_states, states).mean()

        if self.use_discriminator:
            # TODO: add discriminator loss
            disc_factor = self.adopt_weights(self.disc_factor, )
        
            g_loss = - torch.mean(disc_fake)
            v_lambda = self.calculate_lambda(reconstruction_loss, g_loss)
            vq_loss = reconstruction_loss + q_loss + disc_factor * v_lambda * g_loss
            
            d_loss_real = torch.mean(F.relu(1.0 - disc_real))
            d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
            
            gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
        else:
            vq_loss = reconstruction_loss + q_loss
            gan_loss = 0
        return vq_loss, gan_loss
        
    def encode(self, x):
        encoded_states = self.encoder(x)
        quant_states, codebook_indices, q_loss = self.codebook(encoded_states)
        return quant_states, codebook_indices, q_loss
    
    def decode(self, z):
        decoded_states = self.decoder(z)
        return decoded_states

    def calculate_lambda(self, loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        loss_grads = torch.autograd.grad(loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        ratio = torch.norm(loss_grads) / torch.norm(gan_loss_grads + 1e-8)
        ratio = torch.clamp(ratio, 0.0, 1e4).detach()
        return ratio 
    
    @staticmethod
    def adopt_weights(disc_factor, i, treshold, value=0.):
        if i < treshold:
            disc_factor = value
        return disc_factor
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")


class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        else:
            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x


class SQEmbedding(nn.Module):
    '''
    SQ-Embedding instead of default codebook embedding from SQ-VAE paper 2022
    Not tested perfectly!
    '''
    def __init__(
        self, 
        param_var_q, 
        n_embeddings, 
        embedding_dim
    ):
        super(SQEmbedding, self).__init__()
        self.param_var_q = param_var_q
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.normal_()
        self.register_parameter("embedding", nn.Parameter(embedding))
        
        # self.jitter = Jitter(jitter)
        
        log_var_q_scalar = torch.Tensor(1)
        log_var_q_scalar.fill_(10.0).log_()
        self.register_parameter("log_var_q_scalar", nn.Parameter(log_var_q_scalar))

    def encode(self, x, log_var_q):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        if self.param_var_q == "gaussian_1":
            log_var_q_flat = log_var_q.reshape(1, 1)
        elif self.param_var_q == "gaussian_3":
            log_var_q_flat = log_var_q.reshape(-1, 1)
        elif self.param_var_q == "gaussian_4":
            log_var_q_flat = log_var_q.reshape(-1, D)
        else:
            raise Exception("Undefined param_var_q")

        x_flat = x_flat.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * ((embedding - x_flat) ** 2), dim=1)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices

    def forward(self, x, temperature=1):
        M, D = self.embedding.size()
        
        old_x = x
        x = x.view(-1, self.embedding_dim)
        # compute the log var q value
        if self.param_var_q == "gaussian_1":
            log_var_q = self.log_var_q_scalar
        elif self.param_var_q == "gaussian_3" or self.param_var_q == "gaussian_4":
            log_var_q = x[:, self.embedding_dim:] + self.log_var_q_scalar
        else:
            raise Exception("Undefined param_var_q")
        
        batch_size, sample_size = x.size()
        # x_flat = x.reshape(-1, D)
        if self.param_var_q == "gaussian_1":
            log_var_q_flat = log_var_q.reshape(1, 1)
        elif self.param_var_q == "gaussian_3":
            log_var_q_flat = log_var_q.reshape(-1, 1)
        elif self.param_var_q == "gaussian_4":
            log_var_q_flat = log_var_q.reshape(-1, D)
        else:
            raise Exception("Undefined param_var_q")

        x = x.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * (embedding - x) ** 2, dim=1)

        indices = torch.argmin(distances.float(), dim=-1)

        logits = -distances

        encodings = self._gumbel_softmax(logits, tau=temperature, dim=-1)
        quantized = torch.matmul(encodings, self.embedding)
        # quantized = quantized.view_as(x)

        # logits = logits.view(batch_size, sample_size, M)
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)

        precision = torch.exp(-log_var_q)
        # loss = torch.mean(0.5 * torch.sum(precision * (old_x - quantized) ** 2) #,  dim=(1, 2))
        #                   + torch.sum(probabilities * log_probabilities)) #, dim=(1, 2)))
        loss = torch.mean(0.5 * torch.sum(precision * (old_x - quantized) ** 2)) \
        + torch.sum(probabilities * log_probabilities)
        
        encodings = F.one_hot(indices, M).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, indices, loss, perplexity

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, dim=-1):
        eps = torch.finfo(logits.dtype).eps
        gumbels = (
            -((-(torch.rand_like(logits).clamp(min=eps, max=1 - eps).log())).log())
        )  # ~Gumbel(0,1)
        gumbels_new = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels_new.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparameterization trick.
            ret = y_soft

        return ret