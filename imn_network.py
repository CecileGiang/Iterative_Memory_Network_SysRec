# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 14:28:24 2022

@author: Cécile GIANG
"""

import torch
import torch.nn as nn
from functools import partial
from dataloader import *
from multiattention import *


###############################################################################
# -------------------------- CREATION DES DATASETS -------------------------- #
###############################################################################


BATCH_SIZE = 16
ds = MoviesDataset(movies_df)

## Dictionnaires pour l'indexation des items et des timestamps
item2int = ds.item_map
int2item = { value : key for key, value in item2int.items() }
time2int = ds.timestamp_map

## Variables globales
ITEM_SIZE = len(item2int)
TIME_SIZE = len(time2int)
POS_SIZE = ds.pos_size

# Split des données en bases de d'apprentissage et de test
train, test = train_test_split(ds, test_size=0.2)
data_train = DataLoader(train, collate_fn = partial(pad_collate, pos_size = POS_SIZE), shuffle = True, batch_size = BATCH_SIZE, drop_last=True)
data_test = DataLoader(test, collate_fn = partial(pad_collate, pos_size = POS_SIZE), shuffle = True, batch_size = BATCH_SIZE, drop_last=True)


###############################################################################
# ------------------------------- IMN NETWORK ------------------------------- #
###############################################################################


def IMN_net(train, test, embed_size, heads, mem_iter, n_epochs = 20, lr = 0.001):
    
    # Création des embeddings
    embedding = EmbeddingLayer(ITEM_SIZE, TIME_SIZE, POS_SIZE, embed_size) # GLOBAL
    print('item size: ', ITEM_SIZE)
    print('time size: ', TIME_SIZE)
    print('pos size + 1: ', POS_SIZE)
    # Initialisation du module GRU pour le module Memory Update
    gru_mem_up = nn.GRUCell(embed_size, embed_size) #GLOBAL
    
    # Initialisation des modules GRU et Linear pour le module Memory Enhancement
    gru_mem_en = nn.GRUCell(2 * embed_size, embed_size) #GLOBAL
    linear = nn.Linear(embed_size, embed_size)
    
    # Initialisation de la Sequential finale
    sequential = nn.Sequential(
                    nn.Linear(POS_SIZE + embed_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2)
                    #nn.Softmax()
                )
    
    # Création de la liste des modules MultiAttention
    multi_attention = [ MultiAttention(embed_size, heads = heads) for i in range(mem_iter) ]
    
    ma_parameters = []
    for ma in multi_attention:
        ma_parameters += ma.parameters()
    
    parameters = list(embedding.parameters()) + list(gru_mem_up.parameters()) + list(gru_mem_en.parameters()) + list(linear.parameters()) + \
                 list(sequential.parameters()) + ma_parameters
    
    opti = torch.optim.Adam(parameters, lr = lr)
    loss = torch.nn.CrossEntropyLoss(ignore_index = 0)
    
    for epoch in range(n_epochs):
        
        train_loss  = []
        
        for seq_items, time, pos, target_items, label in train:
            
            # Remise à zéro de l'optimiseur
            opti.zero_grad()
            
            # Embedding du batch
            seq_emb, time_emb, pos_emb, target_emb = embedding.forward(seq_items, time, pos, target_items)
            seq_emb = seq_emb + time_emb + pos_emb
            
            # Initialisation de la mémoire
            memory = torch.clone(target_emb)
            
            # Memory Update
            for i in range(mem_iter):
                output, att_vect = multi_attention[i].forward(seq_emb, target_emb, memory)
                memory = gru_mem_up(output, memory)
            
            # Memory Enhancement
            for i in range(3):
                out_lin = linear(memory)
                concat_layer = torch.cat((out_lin, target_emb), dim = -1)
                memory = gru_mem_en(concat_layer, memory)
            
            # ATTENTION: PEUT-ÊTRE QU'IL MANQUE UNE FC
            att_mem = torch.cat((att_vect.squeeze(), memory), dim = -1)
            print(att_vect.squeeze().shape)
            
            print(att_mem.shape)
            final_out = sequential.forward(att_mem)
            
            train_loss.append(loss(final_out, l))
            train_loss[-1].backward()
            opti.step()
        
        train_loss_batch = np.mean(train_loss)
        print("Epoch {} | Train loss = {}" . format(epoch, train_loss_batch))
"""


def IMN_net(seq_items, time, pos, target_items, embed_size=10, heads=1, mem_iter=3):

    
    # Création des embeddings
    embedding = EmbeddingLayer(ITEM_SIZE, TIME_SIZE, POS_SIZE, embed_size) # GLOBAL
    
    # Initialisation du module GRU pour le module Memory Update
    gru_mem_up = nn.GRUCell(embed_size, embed_size) #GLOBAL
    
    # Initialisation des modules GRU et Linear pour le module Memory Enhancement
    gru_mem_en = nn.GRUCell(2 * embed_size, embed_size) #GLOBAL
    linear = nn.Linear(embed_size, embed_size)
    
    # Initialisation de la Sequential finale
    sequential = nn.Sequential(
                    nn.Linear(POS_SIZE + embed_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2),
                    nn.Softmax(dim = 1)
                )
    
    # Embedding du batch
    seq_emb, time_emb, pos_emb, target_emb = embedding.forward(seq_items, time, pos, target_items)
    seq_emb = seq_emb + time_emb + pos_emb
    
    # Initialisation de la mémoire
    memory = torch.clone(target_emb)
    
    # Création de la liste des modules MultiAttention
    multi_attention = [ MultiAttention(embed_size, heads) for i in range(mem_iter) ]
    
    # Memory Update
    for i in range(mem_iter):
        output, att_vect = multi_attention[i].forward(seq_emb, target_emb, memory)
        memory = gru_mem_up(output, memory)
    
    # Memory Enhancement
    for i in range(3):
        out_lin = linear(memory)
        concat_layer = torch.cat((out_lin, target_emb), dim = -1)
        memory = gru_mem_en(concat_layer, memory)
    
    print(att_vect.shape)
    print(memory.shape)
    # ATTENTION: PEUT-ÊTRE QU'IL MANQUE UNE FC
    att_mem = torch.cat((att_vect.squeeze(), memory), dim = -1)
    print(att_mem.shape)
    
    final_out = sequential.forward(att_mem)
    print(final_out)
    print(final_out.shape)

        
## Test
IMN_net(x, t, p, y, embed_size=10, heads=1, mem_iter=10)
        
        """
        
        
IMN_net(data_train, data_test, embed_size=10, heads=1, mem_iter=10)
        
        
        
        
# PARAMETRES DU MODULE, EST QU'UN MODULE POSSEDE UNE FONCTION PARAMETERS
        
        
        
        
        
        
        
        