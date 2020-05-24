import numpy as np
import pandas as pd
import torch
import re
import os, sys

from torch.utils.data import Dataset
from skimage import io, util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Problem_Dataset(Dataset):

    def __init__(self, data_csv_path,  config):
        self.df = pd.read_csv(data_csv_path)
        self.config = config

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        heurstic_result = np.array(self.df.iloc[idx]['plan length'])
        res = torch.from_numpy(heurstic_result).cuda()
        res = res.to(device)

        table = np.load(self.df.iloc[idx]['table'])
        table = table.astype('float32')

        return table, res


