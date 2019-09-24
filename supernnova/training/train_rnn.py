import json
import numpy as np
from tqdm import tqdm
from time import time
from pathlib import Path
from ..utils import training_utils as tu
from ..utils import logging_utils as lu

import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size):
        super(VanillaRNN, self).__init__()

        # Define layers
        self.rnn_layer = torch.nn.LSTM(
            input_size,
            128,
            num_layers=2,
            dropout=0,
            bidirectional=True,
            batch_first=True,
        )
        # self.output_class_layer = torch.nn.Linear()
        # # regression does not use mean vs standard outputs
        self.output_peak_layer = torch.nn.Linear(2 * 128, 1)

    def forward(self, x, x_mask):

        # x is (B, L, D)
        # x_mask is (B, L)

        lengths = x_mask.sum(dim=-1).long()

        # Pack
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        x_packed, hidden = self.rnn_layer(x_packed)

        # unpack
        x_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        # x_padded is (B, L, D)

        x_pred = self.output_peak_layer(x_padded)

        return x_pred


def get_data_batch(list_data, batch_idxs, settings):

    start, end = batch_idxs[0], batch_idxs[-1] + 1

    data = list_data[start:end]

    batch_size = len(batch_idxs)
    input_size = data[0][0].shape[-1]
    max_length = max([len(x[0]) for x in data])

    X = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
    Y = np.zeros((batch_size, max_length, 1), dtype=np.float32)
    lengths = np.zeros((batch_size,), dtype=np.int64)

    for pos, (x, target_tuple, _) in enumerate(data):

        X[pos, : len(x), :] = x.astype(np.float32)
        Y[pos, : len(x), 0] = target_tuple[1].astype(np.float32)
        lengths[pos] = len(x)

    X = torch.from_numpy(X).to(DEVICE)
    Y = torch.from_numpy(Y).to(DEVICE)
    lengths = torch.from_numpy(lengths).to(DEVICE)

    X_mask = (
        torch.arange(max_length).view(1, -1).to(DEVICE) < lengths.view(-1, 1)
    ).float()

    return X, Y, X_mask


def train(settings):
    """Train RNN models with a decay on plateau policy

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    # save training data config
    save_normalizations(settings)

    # Data
    list_data_train, list_data_val = tu.load_HDF5(settings, test=False)
    # Model specification
    rnn = VanillaRNN(len(settings.training_features))
    criterion = nn.CrossEntropyLoss()
    optimizer = tu.get_optimizer(settings, rnn)

    # Prepare for GPU if required
    if settings.use_cuda:
        rnn.cuda()
        criterion.cuda()

    # Keep track of losses for plotting
    loss_str = ""
    d_monitor_train = {"epoch": [], "reg_MSE": []}
    d_monitor_val = {"epoch": [], "reg_MSE": []}

    lu.print_green("Starting training")

    best_loss = float("inf")

    training_start_time = time()

    for epoch in range(settings.nb_epoch):

        num_elem = len(list_data_train)
        num_batches = num_elem // min(num_elem // 2, settings.batch_size)
        list_batches = np.array_split(np.arange(num_elem), num_batches)
        np.random.shuffle(list_batches)

        batch_losses = []

        for batch_idxs in list_batches:

            # Sample a batch in packed sequence form
            X, Y, X_mask = get_data_batch(list_data_train, batch_idxs, settings)
            # Train step : forward backward pass

            Y_pred = rnn(X, X_mask)

            # Loss
            Y = Y.view(-1)
            Y_pred = Y_pred.view(-1)
            X_mask = X_mask.view(-1)
            # Y, Y_pred, X_mask all are (B * L)
            losspeak = ((Y_pred - Y).pow(2) * X_mask).sum() / X_mask.sum()

            rnn.zero_grad()
            losspeak.backward()
            optimizer.step()

            batch_losses.append(losspeak.item())

        loss = np.mean(batch_losses)
        d_monitor_train["reg_MSE"].append(loss)

        # VALID
        num_elem = len(list_data_val)
        num_batches = num_elem // min(num_elem // 2, settings.batch_size)
        list_batches = np.array_split(np.arange(num_elem), num_batches)
        np.random.shuffle(list_batches)

        batch_losses = []

        for batch_idxs in list_batches:

            with torch.no_grad():

                # Sample a batch in packed sequence form
                X, Y, X_mask = get_data_batch(list_data_val, batch_idxs, settings)
                # Train step : forward backward pass

                Y_pred = rnn(X, X_mask)

            # Loss
            Y = Y.view(-1)
            Y_pred = Y_pred.view(-1)
            X_mask = X_mask.view(-1)
            # Y, Y_pred, X_mask all are (B * L)
            losspeak = ((Y_pred - Y).pow(2) * X_mask).sum() / X_mask.sum()

            batch_losses.append(losspeak.item())

        loss = np.mean(batch_losses)
        d_monitor_val["reg_MSE"].append(loss)

        print()
        print("Train", d_monitor_train["reg_MSE"][-1])
        print("Val", d_monitor_val["reg_MSE"][-1])

    #     if (epoch + 1) % settings.monitor_interval == 0:

    #         # Get metrics (subsample training set to same size as validation set for speed)
    #         d_losses_train = tu.get_evaluation_metrics(
    #             settings, list_data_train, rnn, sample_size=len(list_data_val)
    #         )
    #         d_losses_val = tu.get_evaluation_metrics(
    #             settings, list_data_val, rnn, sample_size=None
    #         )

    #         # Add current loss avg to list of losses
    #         for key in d_losses_train.keys():
    #             d_monitor_train[key].append(d_losses_train[key])
    #             d_monitor_val[key].append(d_losses_val[key])
    #         d_monitor_train["epoch"].append(epoch + 1)
    #         d_monitor_val["epoch"].append(epoch + 1)

    #         # Prepare loss_str to update progress bar
    #         loss_str = tu.get_loss_string(d_losses_train, d_losses_val)

    #         tu.plot_loss(d_monitor_train, d_monitor_val, epoch, settings)
    #         if d_monitor_val["loss"][-1] < best_loss:
    #             best_loss = d_monitor_val["loss"][-1]
    #             torch.save(
    #                 rnn.state_dict(),
    #                 f"{settings.rnn_dir}/{settings.pytorch_model_name}.pt",
    #             )

    # lu.print_green("Finished training")

    # training_time = time() - training_start_time

    # tu.save_training_results(settings, d_monitor_val, training_time)


def save_normalizations(settings):
    """Save normalization used for training

    Saves a json file with the normalization used for each feature

    Arguments:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    dic_norm = {}
    for i, f in enumerate(settings.training_features_to_normalize + ["peak"]):
        dic_norm[f] = {}
        for j, w in enumerate(["min", "mean", "std"]):
            dic_norm[f][w] = float(settings.arr_norm[i, j])
    fname = f"{Path(settings.rnn_dir)}/data_norm.json"
    with open(fname, "w") as f:
        json.dump(dic_norm, f, indent=4, sort_keys=True)
