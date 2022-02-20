# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:07:39 2022

@author: Cécile GIANG, KHALFAT Célina, ZHUANG Pascal
"""

###############################################################################
# ----------------------- IMPORTATION DES LIBRAIRIES ------------------------ #
###############################################################################


import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random as rd


###############################################################################
# --------------------------- FONCTIONS AUXILIAIRES ------------------------- #
###############################################################################


def shift_seq (liste, seq_length, decalage) :
    """ Fonction permettant de générer des sous-séquences de liste de longueur seq_length,
        et décalées de decalage.
        @param liste: torch.Tensor, séquence utilisateur
        @param seq_length: int, longueur des sous-séquences
        @param decalage: int, pas de décalage
    """
    seqs = [ liste[i:(i+seq_length)] for i in range(0, len(liste)-seq_length, decalage)]
    seqs.append( liste[-seq_length:])
    return seqs

def augmentation_data (df, seq_length, decalage) :
    data_2 = list()
    for index, rows in df.iterrows():
        seqs_item = shift_seq(rows['item'], seq_length, decalage)
        seqs_timestamp = shift_seq(rows['timestamp'], seq_length, decalage)
        for i in range(len(seqs_item)) :
            data_2.append({'user' : rows['user'] , 'item' : seqs_item[i] , 'timestamp' : seqs_timestamp[i]})
    return pd.DataFrame(data_2)


###############################################################################
# -------------------------- CHARGEMENT DES DATASETS ------------------------ #
###############################################################################


# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Chargement des données de csv à pandas
#movies_df = pd.read_csv("data/ratings_Movies_and_TV.csv", names=["user", "item", "rating", "timestamp"])
books_df = pd.read_csv("../data/ratings_Books.csv", names=["user", "item", "rating", "timestamp"])


class AmazonDataset(Dataset):
    """ Constucteur du dataset pour rating_Movies_and_TV.csv.
    """
    def __init__(self, data, seq_length = 40, decalage = 5):
        """ Constructeur de la classe MoviesDataset.
            @param data: données ratings_Movies_and_TV.csv sous la forme d'un dataframe pandas
            @param seq_length: int, longueur des sous-séquences
            @param decalage: int, pas de décalage
        """
        users, counts = np.unique(data['user'], return_counts=True)
        self.data = data.loc[data['user'].isin(users[counts > seq_length])]
        self.data = self.data.sort_values("timestamp",ascending=True)

        # Map users and items
        self.user_map = {user:num for num,user in enumerate(self.data["user"].unique())}
        self.item_map = {item:num for num,item in enumerate(self.data["item"].unique(), 1)}

        ## Number of users & items
        num_users = len(self.user_map)
        num_items = len(self.item_map)

        self.data["user"] = self.data["user"].map(self.user_map)
        self.data["item"] = self.data["item"].map(self.item_map)

        # Map timestamps into exponentially increasing intervals
        tst_sorted = sorted( self.data['timestamp'].unique() )
        k = int(np.ceil(np.log2(len(tst_sorted))))

        self.timestamp_map = dict()
        current_k = 0
        i = 0

        while current_k <= k and i < len(tst_sorted):
            if i >= 2**current_k:
                current_k += 1
            self.timestamp_map[tst_sorted[i]] = current_k + 1
            i += 1

        self.data["timestamp"] = self.data["timestamp"].map(self.timestamp_map)

        self.data = self.data.groupby('user').agg({'item' : list, 'timestamp' : list}).reset_index()
        self.data = augmentation_data (self.data, seq_length, decalage)

        self.data['label'] = np.where(self.data.index % 2 == 0, 0, 1)


        l = len(list(self.item_map.values())) # Nombre d'items
        dic = self.data.to_dict('records')[::2] # i%2==0

        for d in dic :  
          r = rd.randint(1, l) # Item aléatoire
                    
          while (r == d['item'][-1]): 
              r = rd.randint(1, l)
                        
          d['item'][-1] = r
        self.dataset = self.data

        # Calcul de la taille de séquence maximale
        self.pos_size = max([ len(d[1]) for i, d in self.dataset.iterrows() ])

    def __len__(self):
        """ Retourne la taille du dataset.
        """
        return len(self.dataset)

    def __getitem__(self, i):
        """ Retourne l'échantillon à la position i.
        """
        return torch.tensor( self.dataset.iloc[i]['item'][:-1] ), torch.tensor( self.dataset.iloc[i]['timestamp'][:-1] ), torch.tensor( self.dataset.iloc[i]['item'][-1] ), torch.tensor( self.dataset.iloc[i]['label'] )


def pad_collate(batch, pos_size):
    """ Padding pour les séquences utilisateurs.
    """
    (xx, tt, yy, ll) = zip(*batch)

    # Padding pour les séquences d'items et les séquences de temps
    xx_pad = torch.zeros(len(xx), pos_size, dtype=torch.int)
    tt_pad = torch.zeros(len(tt), pos_size, dtype=torch.int)

    for i in range(len(xx)):
        xx_pad[i] = torch.cat((xx[i], torch.zeros(pos_size - len(xx[i]))))
        tt_pad[i] = torch.cat((tt[i], torch.zeros(pos_size - len(tt[i]))))

    # Padding des positions des items dans une séquence
    pp_pad = xx_pad.numpy().copy()

    for i in range(len(pp_pad)):
        mask = np.where(pp_pad[i], 1, 0)
        pp_pad[i] = np.arange(1, len(pp_pad[i]) + 1) * mask

    return xx_pad, tt_pad, torch.tensor(pp_pad), torch.stack(list(yy), dim=0), torch.stack(list(ll), dim=0)