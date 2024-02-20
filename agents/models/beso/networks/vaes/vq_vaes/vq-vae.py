
import torch
import numpy as np
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from .v_quantizer_layer import VectorQuantizer, VectorQuantizerEMA


class VQVAE(nn.Module):
    
    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        decay: float,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        
    ) -> None:
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder)

        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)

        self.decoder = hydra.utils.instantiate(decoder)

    def forward(self, x):
        
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity