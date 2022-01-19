# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from dataloader import *

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
        
        self.item_embedding = nn.Embedding(self.item_size +1, self.embed_dim, padding_idx = 0)
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
    """ Classe qui effectue le Multi-Way Attention.
    """
    def __init__(self, embed_dim = 10, heads = 1, mask=False):
        """ @param embed_dim : Dimension des embeddings
            @param heads : Nombre de head
        """
        super(MultiAttention, self).__init__()
               
        self.heads = heads
        self.embed_dim = embed_dim
        self.mask = mask
        
        # Linear layers
        self.keys = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        self.values = nn.Linear(embed_dim, embed_dim * heads, bias=False)
        
        # Unify Linear layer
        self.unifyheads = nn.Linear(embed_dim * heads, embed_dim)

        # 2 couches fully-connected
        self.fully_connected = nn.Sequential(
                                    nn.Linear(self.embed_dim * 4, 8),
                                    nn.Sigmoid(),
                                    nn.Linear(8, 1),
                                    nn.Sigmoid())
        
        
    def forward(self, seq_items, target_items, memory):
        """ @param seq_items: torch.tensor, batch de séquences utilisateurs de taille batch_size x length
            @param target_items: torch.tensor, tenseur des items cibles de taille batch_size
            @param memory: torch.tensors, mémoire
        """
        # Keys, queries, values
        if self.heads == 1 :
            self.e = self.keys(seq_items)
            self.v = self.queries(target_items)
            self.m = memory
            self.V_ = self.values(seq_items)
            
        else :
            # Get number of training examples in a batch
            N = target_items.shape[0] 
            
            value_len, key_len, query_len = seq_items.shape[1], seq_items.shape[1], 1 # target_items.shape[1]
        
            # Split into self.heads different pieces
            self.e = self.keys(seq_items)      .view(N, key_len, self.heads, self.embed_dim)
            self.V_ = self.values(seq_items)   .view(N, value_len, self.heads, self.embed_dim)
            self.v = self.queries(target_items).view(N, self.heads, self.embed_dim)
            self.m = memory                    .view(N, 1, self.embed_dim)
        
        # On change l'ordre des dimensions de self.e
        if self.heads == 1 :
            e_ = self.e.permute((1,0,2))
        else :
            e_ = self.e.permute((1,0,2,3)) # (key_len, N, heads, embed_dim)
        
        # Masque sur e pour garder le padding à 0
        self.e_mask = np.where(e_.cpu().detach().numpy(), 1, 0)
        
        # Similarités : Euclidean distance and Hadamard distance
        a1 = torch.Tensor((self.v - e_).cpu().detach().numpy() * self.e_mask) # (key_len, N, heads, embed_dim)
        a2 = torch.Tensor((self.v * e_).cpu().detach().numpy() * self.e_mask) # (key_len, N, heads, embed_dim)
        a3 = torch.Tensor((self.m - e_).cpu().detach().numpy() * self.e_mask) # (key_len, N, heads, embed_dim)
        a4 = torch.Tensor((self.m * e_).cpu().detach().numpy() * self.e_mask) # (key_len, N, heads, embed_dim)
            
        # Concatenation des similarités
        alpha = torch.cat((a1, a2, a3, a4), dim = -1).to(device)
        # (key_len, N, heads, embed_dim * 4)
        
        # Permutation des dimensions : N et key_len
        if self.heads == 1 :
            alpha = alpha.permute((1,0,2))
        else :
            alpha = alpha.permute((1,0,2,3)) # (N, key_len, heads, embed_dim * 4)            
        
        # Calcul du vecteur d'attention
        self.att_vect = self.fully_connected(alpha) # (N, key_len, heads, query_len)
        
        # Output : (N, query_len, embed_dim)
        if self.heads == 1 :
            self.output = torch.matmul(self.att_vect.permute((0,2,1)), self.V_).squeeze()
        else :
            out = torch.einsum("nlhq,nlhd->nqhd", [self.att_vect, self.V_]).view(
                N, self.heads * self.embed_dim
            )
            self.output = self.unifyheads(out)
        
            
        return self.output, self.att_vect
        
    
    def parameters(self):
        return list(self.keys.parameters()) + list(self.queries.parameters()) \
            + list(self.values.parameters()) + list(self.fully_connected.parameters()) \
            + list(self.unifyheads.parameters())
    
# Tests

# from functools import partial

# BATCH_SIZE = 16
# ds = MoviesDataset(movies_df)

# ## Dictionnaires pour l'indexation des items et des timestamps
# item2int = ds.item_map
# int2item = { value : key for key, value in item2int.items() }
# time2int = ds.timestamp_map

# ## Variables globales
# ITEM_SIZE = len(item2int)
# TIME_SIZE = len(time2int)
# POS_SIZE = ds.pos_size

# # Split des données en bases de d'apprentissage et de test
# train, test = train_test_split(ds, test_size=0.2)
# data_train = DataLoader(train, collate_fn = partial(pad_collate, pos_size = POS_SIZE), shuffle = True, batch_size = BATCH_SIZE, drop_last=True)
# data_test = DataLoader(test, collate_fn = partial(pad_collate, pos_size = POS_SIZE), shuffle = True, batch_size = BATCH_SIZE, drop_last=True)

# x, t, p, y, l = next(iter(data_train))
# el = EmbeddingLayer(item_size = len(item2int), time_size = len(time2int), pos_size = len(x[0]), embed_dim = 10)
# s_e, t_e, p_e, v = el.forward(x, t, p, y)
# m_e = torch.clone(v)
# ma = MultiAttention(embed_dim = 10, heads = 2)
# ma.forward(s_e, v, m_e)

# print("Att_vec shape :", ma.att_vect.shape)
# print("Values shape :", ma.V_.shape)
# print("Output shape :", ma.output.shape)