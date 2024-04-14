import math
import torch
import torch.nn as nn 
import numpy as np
from mmcv.cnn import bias_init_with_prob, xavier_init


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class MotionMLP(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=512, identity=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.identity = identity

        multires = 10
        embed_kwargs = {
                'include_input' : True,
                'input_dims' : c_dim,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
        }
        self.pos_embedder = Embedder(**embed_kwargs)

        self.fc = nn.Sequential(
            nn.Linear(f_dim + self.pos_embedder.out_dim, 2*f_dim),
            nn.LayerNorm(2*f_dim),
            nn.ReLU(),
            nn.Linear(2*f_dim, f_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.fc:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            

    def forward(self, x, pose_info):
        pose_embed = self.pos_embedder.embed(pose_info)
        xc = torch.cat([x, pose_embed], dim=-1)
        out = self.fc(xc)

        if self.identity:
            out = out + x
        
        return out
