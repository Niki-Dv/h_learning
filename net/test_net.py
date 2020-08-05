import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import logging
import time

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



def test__best(net_path, test_loader):

    logger.info(f'\nBest Model Test Initiated.')
    print(net_path)
    net = torch.load(net_path)
    correct = 0
    distance_count = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs.to(device))
            outputs = outputs.view(-1)
            logger.info("Labels: {}".format(labels))
            logger.info("Outputs: {}".format(outputs))
            for i, idx in enumerate(outputs):
                total += 1
                distance_count += torch.abs((outputs[i] - labels[i]).cpu())
                if (torch.round(outputs[i].cpu()) - labels[i]) == 0:
                    correct += 1
        acc_round = correct / total
        dist_avg = distance_count/ total
    logger.info(f'\nBest Model Accuracy with round: %{round(acc_round * 100)}.')
    logger.info(f"\nBest model avergae distance: {dist_avg}")


if __name__ == '__main__':
    NET_TO_TEST_PATH = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\net_results\saved_models\no-fft.pt"
    DATA_CSV_PATH = r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\Data_generator\Final_data\Rovers_2_27755.csv"
    prob_data = Problem_Dataset(DATA_CSV_PATH, config)
    _, _, test_dataset = torch.utils.data.random_split(prob_data,[0, 0,len(prob_data)])
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    test__best(NET_TO_TEST_PATH, test_loader)





























