# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:08:11 2022

@author: Cécile GIANG, KHALFAT Célina, ZHUANG Pascal
"""

from dataloader import *
from multiattention import *

import torch
import torch.nn as nn
from functools import partial
from sklearn.metrics import accuracy_score, roc_curve, auc
from scipy.special import softmax
from torchsummary import summary


###############################################################################
# ------------------------------- UTILITAIRE -------------------------------- #
###############################################################################


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    return total_params


###############################################################################
# -------------------------- CREATION DES DATASETS -------------------------- #
###############################################################################


BATCH_SIZE = 256
ds = AmazonDataset(books_df, seq_length = 1000, decalage = 5)

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


class IMN (nn.Module):
    """ Module modélisant le réseau IMN global.
    """
    def __init__(self, train, test, embed_size, heads, mem_iter, p):
        """ @param train: torch.Tensor, données d'entraînement
            @param test: torch.Tensor, données de test
            @param embed_size: int, taille de l'espace des représentations des items
            @param heads: int, nombre de heads pour le calcul de l'attention
            @param mem_iter: int, nombre de mises à jours de la mémoire (Memory Update Module)
            @param p: int, probabilité de négliger un neurone pour le Dropout
        """
        
        super(IMN, self).__init__()

        # Création des embeddings
        self.embedding = EmbeddingLayer(ITEM_SIZE, TIME_SIZE, POS_SIZE, embed_size).to(device)

        # Initialisation du module GRU pour le module Memory Update
        self.gru_mem_up = nn.GRUCell(embed_size, embed_size).to(device)

        # Initialisation des modules GRU et Linear pour le module Memory Enhancement
        self.gru_mem_en = nn.GRUCell(2 * embed_size, embed_size).to(device)
        self.linear = nn.Linear(embed_size, embed_size).to(device)

        # Initialisation de la Sequential finale
        self.sequential = nn.Sequential(
                        #nn.BatchNorm1d(embed_size * 2),
                        # nn.Linear(POS_SIZE + embed_size, 512),
                        #nn.Linear(embed_size * 2, 512),
                        #nn.ReLU(),
                        #nn.BatchNorm1d(512),
                        #nn.Dropout(),
                        nn.Linear(embed_size*2, 128),
                        nn.ReLU(),
                        nn.BatchNorm1d(128),
                        nn.Dropout(p=p),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.BatchNorm1d(64),
                        #nn.Dropout(p=p),
                        nn.Linear(64, 2)

                    ).to(device)

        # Création de la liste des modules MultiAttention
        self.mem_iter = mem_iter
        self.multi_attention = [ MultiAttention(embed_size, heads = heads).to(device) for i in range(self.mem_iter) ]

    def forward (self,seq_items, time, pos, target_items, label) :
        """ @param seq_items: torch.Tensor, batch des séquences utilisateurs
            @param time: torch.Tensor, batch des temps associés à chaque item de chaque séquence
            @param pos: torch.Tensor, batch des positions associées à chaque item dans une séquence
            @param target_items: torch.Tensor, batch des items targets
            @param label: torch.Tensor, batch des classes (1 si l'item est bien cliqué au prochain pas, 0 sinon)
        """
        # On met les données sur gpu
        seq_items = seq_items.to(device)
        time = time.to(device)
        pos = pos.to(device)
        target_items = target_items.to(device)
        label = label.to(device)

        # Embedding du batch
        seq_emb, time_emb, pos_emb, target_emb = self.embedding.forward(seq_items, time, pos, target_items)
        seq_emb = seq_emb + time_emb + pos_emb

        # Initialisation de la mémoire
        memory = torch.clone(target_emb).to(device)

        # Memory Update
        for i in range(self.mem_iter):
            output, att_vect = self.multi_attention[i].forward(seq_emb, target_emb, memory)
            output = output.to(device)
            att_vect = att_vect.to(device)
            memory = self.gru_mem_up(output, memory)

        # Memory Enhancement
        for i in range(3):
            out_lin = self.linear(memory)
            concat_layer = torch.cat((out_lin, target_emb), dim = -1)
            memory = self.gru_mem_en(concat_layer, memory)

        att_mem = torch.cat((target_emb, memory), dim = -1)

        final_out = self.sequential.forward(att_mem)

        return torch.sigmoid(final_out).to(device)

    def get_parameters(self):
        ma_parameters = []
        for ma in self.multi_attention:
            ma_parameters += ma.parameters()

        return list(self.embedding.parameters()) + list(self.gru_mem_up.parameters()) + list(self.gru_mem_en.parameters()) +\
                     list(self.linear.parameters()) + list(self.sequential.parameters()) + ma_parameters


def IMN_net(train, test, embed_size=10, heads=1, mem_iter=3, n_epochs = 40, lr = 0.001, reg=1e-4, p=0.5):
    """ Boucle d'apprentissage d'IMN.
        @param train: torch.Tensor, données d'entraînement
        @param test: torch.Tensor, données de test
        @param embed_size: int, taille de l'espace des représentations des items
        @param heads: int, nombre de heads pour le calcul de l'attention
        @param mem_iter: int, nombre de mises à jours de la mémoire (Memory Update Module)
        @param n_epochs: int, nombre d'époques
        @param lr: float, learning rate
        @param reg: float, taux de régularisation
        @param p: int, probabilité de négliger un neurone pour le Dropout
    """
    # Initialisation du module IMN
    imn = IMN(train, test, embed_size, heads, mem_iter,p)
    
    # Initialisation de l'optimiseur et de la loss
    opti = torch.optim.Adam(imn.get_parameters(), lr = lr, weight_decay=reg)
    loss = torch.nn.CrossEntropyLoss  ()
    
    # Affichage du nombre de paramètres
    print("Total Trainable Params {} | Nb train data = {}" . format( count_parameters(imn), len(train)))
    
    
    for epoch in range(n_epochs):

        imn.train()
        train_loss, test_loss  = [], []
        train_auc, test_auc = [], []
        
        # ------- Phase d'apprentissage
        
        for seq_items, time, pos, target_items, label in train:
            
            # Données sur device
            label = label.to(torch.int64).to(device)
            
            # Remise à zéro de l'optimiseur
            opti.zero_grad()
            final_out = imn.forward(seq_items, time, pos, target_items, label).squeeze()
            
            # Calcul de la loss et back-propagation
            loss_ = loss(final_out, label)
            train_loss.append(loss_.item())
            fpr, tpr, _ = roc_curve(label.cpu().detach().numpy(), softmax(final_out.cpu().detach().numpy())[:,1], pos_label=1 )
            train_auc.append(auc(fpr, tpr))
            
            loss_.backward()
            opti.step()
        
        # ------- Phase de test 
        
        imn.eval()
        
        with torch.no_grad():
            for seq_items_tes, time_test, pos_test, target_items_test, label_test in test :
                
                # Inférence
                final_out_test = imn.forward(seq_items_tes, time_test, pos_test, target_items_test, label_test).squeeze()
                
                # Données sur device
                label_test = label_test.to(torch.int64).to(device)
                
                # Calcul de la loss
                loss_test_ = loss(final_out_test, label_test)
                test_loss.append(loss_test_.item())
                fpr_test, tpr_test, _ = roc_curve(label_test.cpu().detach().numpy(), softmax(final_out_test.cpu().detach().numpy())[:,1], pos_label=1 )
                test_auc.append(auc(fpr_test, tpr_test))

        train_loss_batch = np.mean(train_loss)
        train_auc_batch = np.mean(train_auc)
        test_loss_batch = np.mean(test_loss)
        test_auc_batch = np.mean(test_auc)
        
        print("Epoch {} | Train loss = {:.5} | Train AUC = {:.5} | Test loss = {:.5} | Test AUC = {:.5}" . format(epoch, train_loss_batch, train_auc_batch, test_loss_batch, test_auc_batch))


## Exemple de commande pour exécuter IMN
# IMN_net(data_train, data_test, embed_size=3, heads=1, mem_iter=3, n_epochs = 200, lr = 0.001, reg=1e-6, p=0.2)
