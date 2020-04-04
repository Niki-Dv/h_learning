import numpy as np
import pandas as pd
import torch
import re
import os, sys

from torch.utils.data import Dataset
from skimage import io, util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Problem_Dataset(Dataset):

    def __init__(self, data_csv_path,  config, ftrans= None):
        self.df = pd.read_csv(data_csv_path)
        self.trans = ftrans
        self.config = config

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        heurstic_result = np.array(self.df.iloc[idx]['plan length'])
        res = heurstic_result
        #res = res.to(device)

        img_relative_path = re.match("[\s\S]+Data_generator\/([\s\S]+)", self.df.iloc[idx]['image'])[1]
        problem_image_path = os.path.join(self.config.data_gen_path, img_relative_path)
        image = io.imread(problem_image_path)
        image = util.invert(image)
        image = np.asarray(image)
        image = image.astype('float32')
        image = image / 255.0  # Normalize the data

        if self.trans:
            pass  # todo: add fft2 if needed

        image = torch.from_numpy(image)
        #image = image.to(device)
        image.unsqueeze_(0)

        return image, res



