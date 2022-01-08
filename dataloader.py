###############################################################################
# ----------------------- IMPORTATION DES LIBRAIRIES ------------------------ #
###############################################################################


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


###############################################################################
# -------------------------- CHARGEMENT DES DATASETS ------------------------ #
###############################################################################


# ratings_Movies_and_TV.csv
movies_df = pd.read_csv("../data/ratings_Movies_and_TV.csv", names=["user", "item", "rating", "timestamp"])


class MoviesDataset(Dataset):
    """ Constucteur du dataset pour rating_Movies_and_TV.csv.
    """
    def __init__(self, data, min_length = 1000):
        """ Constructeur de la classe MoviesDataset.
            @param data: données ratings_Movies_and_TV.csv sous la forme d'un dataframe pandas
            @param min_length: int, longueur minimale de l'historique des utilisateurs
        """
        users, counts = np.unique(data['user'], return_counts=True)
        self.data = data.loc[data['user'].isin(users[counts > min_length])]
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
        
    def __len__(self):
        """ Retourne la taille du dataset.
        """
        return len(self.data)
    
    def __getitem__(self, i):
        """ Retourne l'échantillon à la position i.
        """
        return torch.tensor( self.data.iloc[i]['item'][:-1] ), torch.tensor( self.data.iloc[i]['timestamp'][:-1] ), torch.tensor( self.data.iloc[i]['item'][-1] )


def pad_collate(batch):
    """ Padding pour les séquences utilisateurs.
    """
    (xx, tt, yy) = zip(*batch)
    
    # Padding pour les séquences d'items et les séquences de temps
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    tt_pad = pad_sequence(tt, batch_first=True, padding_value=0)
    
    # Padding des positions des items dans une séquence
    pp_pad = xx_pad.numpy().copy()
    
    for i in range(len(pp_pad)):
        mask = np.where(pp_pad[i], 1, 0)
        pp_pad[i] = np.arange(1, len(pp_pad[i]) + 1) * mask
    
    return xx_pad, tt_pad, torch.tensor(pp_pad), torch.tensor(yy)