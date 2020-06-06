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

def test__best(exp_dict, test_loader):

    logger.info(f'\nBest Model Test Initiated.')
    net = torch.load(exp_dict['path_to_model'])
    correct = 0
    distance_count = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs.to(device))
            outputs = outputs.view(-1)
            logger.debug("Labels: {}".format(labels))
            logger.debug("Outputs: {}".format(outputs))
            for i, idx in enumerate(outputs):
                total += 1
                distance_count += torch.abs((outputs[i] - labels[i]).cpu())
                if (torch.round(outputs[i].cpu()) - labels[i]) == 0:
                    correct += 1
        acc_round = correct / total
        dist_avg = distance_count/ total
    logger.info(f'\nBest Model Accuracy with round: %{round(acc_round * 100)}.')
    logger.info(f"\nBest model avergae distance: {dist_avg}")



torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

key_list = ['net_1', 'net_1_double', 'net_1_triple', 'net_2', 'net_2_double',
             'net_2_triple', 'net_3', 'net_3_double', 'net_3_triple']


""" SETUP PARSER """
parser = argparse.ArgumentParser()
parser.add_argument('net_key', type=str, help='the key of the desired net.',
                    choices=key_list)
parser.add_argument('use_ft', type=int, help='use fourier transform on the data.', choices=list(range(2)))
parser.add_argument('-optimizer', type=str, help='what optim to use?', choices=['Adam', 'SGD'], default='Adam')
parser.add_argument('-epochs', type=int, help='ho many epochs to perform', default=1000)
parser.add_argument('-batch', type=int, help='ho many batches', default=8)
parser.add_argument('-lr', type=float, help='learning rate', default=0.00001)
parser.add_argument('-betas', type=float, help='betas for Adam optim', default=(0.9,0.9999))
parser.add_argument('-momentum', type=float, help='momentum for SGD optim', default=0.9)


if __name__ == '__main__':
    args = parser.parse_args()  # Disable during debugging
    config = net_config.config()
    logger = logging.getLogger()
    net_config.define_logger(logger, os.path.join(config.results_dir, "net_log.log"))
    logger.debug("Defined logger")
    logger.info(f"using device: {device}")

    if args.optimizer == 'Adam':
        optimizer = (optim.Adam, {'lr':args.lr, 'betas':args.betas})
    elif args.optimizer == 'SGD':
        optimizer = (optim.SGD, {'lr':args.lr, 'momentum':args.momentum})

    exp_dict = {
        'net_key': args.net_key,
        'fourier': args.use_ft,
        'optimizer': optimizer,
        'max_num_epochs': args.epochs,
        'path_to_model': os.path.join(config.saved_models_path, "no-fft.pt")
    }

    logger.info(f'Starting CNN training...')
    logger.info(f'Options: \n{exp_dict}')

    """ ESTABLISH DATASET """
    prob_data = Problem_Dataset(config.data_csv_path, config)
    dataloader = DataLoader(prob_data, batch_size=args.batch, shuffle=True)

    """ ESTABLISH A NETWORK """
    net_key = exp_dict['net_key']
    net = MLP_1(config.input_size)
    # net = torch.load(
    #     r"C:\Users\NikiDavarashvili\OneDrive - Technion\Desktop\Project\net_results\saved_models\no-fft.pt")
    # net.eval()
    net = net.to(device)

    logger.info(f'CNN established:\n{net}')


    """ ESTABLISH A LOSS FUNCTION """
    criterion = nn.MSELoss()
    criterion.to(device)
    logger.info(f'Loss function established: {criterion}')

    """ ESTABLISH AN OPTIMIZER """
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer, cfg = exp_dict['optimizer']
    optimizer = optimizer(net.parameters(), **cfg)
    logger.info(f'Optimizer established: {criterion}')

    """ TRAIN """
    train_size = int(0.6 * len(prob_data))
    valid_size = int(0.2 * len(prob_data))
    test_size = len(prob_data) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(prob_data,
                                                                               [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)

    best_acc = 0
    logger.info(f'Training Initiated.')

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    best_valid_loss = 20
    patience = 50
    epochs_without_improvement = 0
    #test__best(exp_dict, test_loader)
    for epoch in range(exp_dict['max_num_epochs']):  # loop over the dataset multiple times
        logger.info("On epoch: {}".format(epoch))
        t0 = time.time()
        ###################
        # train the model #
        ###################
        net.train()
        running_loss = 0.0
        for i, sample in enumerate(train_loader):
            inputs, targets = sample
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.to(device))
            res = outputs.clone().type(dtype=torch.float).view(-1)
            #res = res.to(device)
            targets = targets.clone().type(dtype=torch.float)

            loss = criterion(res, targets)
            loss.backward()
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        net.eval()  # prep model for evaluation
        for i, sample in enumerate(valid_loader):
            inputs, targets = sample
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = net(inputs.to(device))
            outputs = outputs.view(-1)
            # calculate the loss
            logger.info("targets: {}".format(targets))
            logger.info("Outputs: {}".format(outputs))
            loss = criterion(outputs, targets)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_time = round(time.time() - t0, 2)

        print_msg = (f'[{epoch}/{exp_dict["max_num_epochs"]}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} ' +
                     f'epoch time: {epoch_time}')

        logger.info(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        logger.info("current best valid loss: {}".format(best_valid_loss))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss * 0.99
            logger.info('New best model, saving ...')
            torch.save(net, exp_dict['path_to_model'])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > patience:
            logger.info("Early stopping")
            logger.info(f"train loss: {avg_train_losses}")
            logger.info(f"valid loss: {avg_valid_losses}")
            break
    """ TEST BEST MODEL """
    test__best(exp_dict, test_loader)

    plt.figure()
    plt.plot(avg_valid_losses, label="validation average MSE Loss")
    plt.plot(avg_train_losses, label="train average MSE Loss")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    save_path = exp_dict['path_to_model'] + "Figure.png"
    plt.savefig(save_path)
    plt.show()
