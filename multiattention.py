# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:12:01 2021

@author: Cécile GIANG
"""


import torch
import torch.nn as nn
from dataloader import *

# Variables globales


class EmbeddingLayer(nn.Module):
    """ Classe pour l'embedding des items des séquences utilisateurs et des items targets.
    """
    def __init__(self, item_size, time_size, pos_size, embed_dim):
        """ @param item_size: int, nombre total d'items, ie len(item2int)
            @param time_size: int, nombre total de timestamps, ie len(time2int)
            @param pos_size: int, nombre total de positions, ie len(x[0])
            @param embed_dim: int, dimension des embeddings
        """
        super(EmbeddingLayer, self).__init__()
        
        self.item_size = item_size
        self.time_size = time_size
        self.pos_size = pos_size
        self.embed_dim = embed_dim
        
        self.item_embedding = nn.Embedding(self.item_size + 1, self.embed_dim, padding_idx = 0)
        self.time_embedding = nn.Embedding(self.time_size + 1, self.embed_dim, padding_idx = 0)
        self.pos_embedding = nn.Embedding(self.pos_size + 1, self.embed_dim, padding_idx = 0)
    
    def forward(self, seq_items, times, pos, target_items):
        """ @param seq_items: torch.tensor, batch de séquences utilisateurs de taille batch_size x length
            @param times: torch.tensors, catégorie temporelle des temps correspondant à chaque item, batch_size x length
            @param pos: torch.tensors, tenseur indiquant pour chaque item sa position dans une séquence, batch_size x length
            @param target_items: torch.tensor, tenseur des items cibles de taille batch_size
        """
        seq_emb = self.item_embedding(seq_items)
        time_emb = self.time_embedding(times)
        pos_emb = self.pos_embedding(pos)
        target_emb = self.item_embedding(target_items)
        
        return seq_emb, time_emb, pos_emb, target_emb


class MultiAttention(nn.Module):
    def __init__(self, embed_dim, heads = 1):
        """ @param seq_items: torch.tensor, batch de séquences utilisateurs de taille batch_size x length
            @param target_items: torch.tensor, tenseur des items cibles de taille batch_size
            @param memory: torch.tensors, mémoire
        """
        super(MultiAttention, self).__init__()
               
        self.heads = heads
        self.embed_dim = embed_dim
        self.heads_dim = self.embed_dim // self.heads
        
        assert (
            self.heads_dim * self.heads == self.embed_dim
        )
        
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        
        # 2 couches fully-connected
        self.fully_connected = nn.Sequential(
                                    nn.Linear(self.embed_dim * 4, 8),
                                    nn.Sigmoid(),
                                    nn.Linear(8, 1),
                                    nn.Sigmoid())
        
    def forward(self, seq_items, target_items, memory):
        
        # Keys, queries, values
        self.e = self.keys(seq_items)
        self.v = self.queries(target_items)
        self.m = memory
        self.V_ = self.values(seq_items)
        
        # On change l'ordre des dimensions de self.e
        e_ = torch.permute(self.e, (1,0,2))
        
        # Masque sur e pour garder le padding à 0
        self.e_mask = np.where(e_.cpu().detach().numpy(), 1, 0)
        
        l_alpha = []
        l_alpha.append((self.v - e_).cpu().detach().numpy() * self.e_mask)
        l_alpha.append((self.v * e_).cpu().detach().numpy() * self.e_mask)
        l_alpha.append((self.m - e_).cpu().detach().numpy() * self.e_mask)
        l_alpha.append((self.m * e_).cpu().detach().numpy() * self.e_mask)
        
        for i in range(len(l_alpha)):
            l_alpha[i] = torch.permute(torch.Tensor(l_alpha[i]), (1,0,2))
        
        # Concatenation des similarités
        self.alpha = torch.cat((l_alpha[0], l_alpha[1], l_alpha[2], l_alpha[3]), dim = -1).to(device)
        
        # Calcul du vecteur d'attention
        self.att_vect = self.fully_connected(self.alpha)
        self.output = torch.matmul(torch.permute(self.att_vect, (0,2,1)), self.V_).squeeze()
        
        return self.output, self.att_vect
    
    def parameters(self):
        return list(self.keys.parameters()) + list(self.queries.parameters()) + list(self.values.parameters()) + list(self.fully_connected.parameters())
        
## Tests
#x, t, p, y = next(iter(data_train))
#el = EmbeddingLayer(item_size = len(item2int), time_size = len(time2int), pos_size = len(x[0]), embed_dim = 10)
#s_e, t_e, p_e, v = el.forward(x, t, p, y)
#m_e = torch.clone(v)
#ma = MultiAttention(s_e, v, m_e)
#ma.forward()