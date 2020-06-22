import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import time
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io, util

from dataset import Problem_Dataset
from architectures import PlaNet, MLP_1

import net_config

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = net_config.config()
logger = logging.getLogger()
net_config.define_logger(logger, os.path.join(config.results_dir, "test_net_log.log"))
logger.debug("Defined logger")
logger.info(f"using device: {device}")

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

        return table, res, idx

def fill_net_heur(data_df, net_path, test_loader):

    data_df["Net Heur"] = -1
    net = torch.load(net_path)
    net.eval()

    total, distance_count, correct = 0, 0, 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels, idx = data
            outputs = net(inputs.to(device))
            outputs = outputs.view(-1)
            print(f"Actual: {labels} and net: {outputs}")
            #data_df.at[idx, "Net Heur"] = outputs.item()

            for i, idx in enumerate(outputs):
                total += 1
                distance_count += torch.abs((outputs[i] - labels[i]).cpu())
                if (torch.round(outputs[i].cpu()) - labels[i]) == 0:
                    correct += 1
    dist_avg = distance_count / total
    print(f"\nBest model avergae distance: {dist_avg}")
    return data_df


if __name__ == '__main__':

    NET_TO_TEST_PATH = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\net_results\saved_models\bigger_4000_128_1.pt"
    DATA_CSV_PATH = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\generated_problems\goal_as_column\csv_dir\info_19_06_2020_12_15_50.csv"
    data_df = pd.read_csv(DATA_CSV_PATH)
    prob_data = Problem_Dataset(DATA_CSV_PATH, config)
    _, _, test_dataset = torch.utils.data.random_split(prob_data,[0, 0,len(prob_data)])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    data_df = fill_net_heur(data_df, NET_TO_TEST_PATH, test_loader)
    data_df.to_csv(DATA_CSV_PATH)




























