import torch
import json
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from time import time
import pandas as pd
from pathlib import Path
import h5py
import supernnova.conf as conf
from supernnova.utils import logging_utils as lu

import matplotlib.pylab as plt

plt.switch_backend("agg")
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm

import torch
import torch.nn.functional as F
from supernnova.utils.visualization_utils import FILTER_COLORS


class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size):
        super(VanillaRNN, self).__init__()

        # Define layers
        self.rnn_layer = torch.nn.LSTM(
            input_size,
            32,
            num_layers=2,
            dropout=0,
            bidirectional=True,
            batch_first=True,
        )
        # self.output_class_layer = torch.nn.Linear()
        # # regression does not use mean vs standard outputs
        self.output_peak_layer = torch.nn.Linear(2 * 32, 1)

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


def data_loader(settings):

    file_name = f"{settings.processed_dir}/database.h5"
    lu.print_green(f"Loading {file_name}")

    with h5py.File(file_name, "r") as hf:

        list_data_train = []
        list_data_val = []

        config_name = f"{settings.source_data}_{settings.nb_classes}classes"

        n_samples = hf["data"].shape[0]
        subset_n_samples = int(n_samples*settings.data_fraction)

        idxs = np.random.permutation(n_samples)[:subset_n_samples]
        idxs = idxs[:]
        idxs_train = idxs[: subset_n_samples // 2]
        idxs_val = idxs[subset_n_samples // 2 :]

        n_features = hf["data"].attrs["n_features"]

        training_features = " ".join(hf["features"][:][settings.idx_features])
        lu.print_green("Features used", training_features)

        list_data_train = []
        for i in tqdm(idxs_train):
            arr = hf["data"][i].reshape(-1, n_features)
            # arr is (length_LC, n_features)
            df = pd.DataFrame(arr, columns=hf["features"][:])
            df["target"] = hf["target_2classes"][i]
            df["PEAKMJDNORM"] = hf["PEAKMJDNORM"][i]
            df["time"] = df["delta_time"].cumsum()
            df["delta_PEAKMJDNORM"] = df["PEAKMJDNORM"] - df["time"]
            list_data_train.append(df)

        list_data_val = []
        for i in tqdm(idxs_val):
            arr = hf["data"][i].reshape(-1, n_features)
            # arr is (length_LC, n_features)
            df = pd.DataFrame(arr, columns=hf["features"][:])
            df["target"] = hf["target_2classes"][i]
            df["PEAKMJDNORM"] = hf["PEAKMJDNORM"][i]
            df["time"] = df["delta_time"].cumsum()
            df["delta_PEAKMJDNORM"] = df["PEAKMJDNORM"] - df["time"]
            list_data_val.append(df)

    return list_data_train, list_data_val


def batch_loop(model, opt, list_data, list_features, grad_enabled=True):

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    input_size = len(list_features)
    batch_size = 500
    n_samples = len(list_data)
    num_batches = max(1, n_samples // batch_size)
    list_batches = np.array_split(np.arange(n_samples), num_batches)

    batch_losses = []

    # Training loop
    for batch_idxs in tqdm(list_batches):
        start, end = batch_idxs[0], batch_idxs[-1] + 1

        data = list_data[start:end]

        batch_size = len(batch_idxs)
        max_length = max(map(len, data))

        X = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
        Y = np.zeros((batch_size, max_length, 1), dtype=np.float32)
        lengths = np.zeros((batch_size,), dtype=np.int64)

        for pos, df in enumerate(data):
            X[pos, : len(df), :] = df[list_features].values.astype(np.float32)
            Y[pos, : len(df), 0] = df["delta_PEAKMJDNORM"].values.astype(np.float32)
            lengths[pos] = len(df)

        torch.set_grad_enabled(grad_enabled)

        X = torch.from_numpy(X).to(device)
        Y = torch.from_numpy(Y).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        X_mask = (
            torch.arange(max_length).view(1, -1).to(device) < lengths.view(-1, 1)
        ).float()

        Y_pred = model(X, X_mask)

        # Loss
        Y = Y.view(-1)
        Y_pred = Y_pred.view(-1)
        X_mask = X_mask.view(-1)
        # Y, Y_pred, X_mask all are (B * L)
        losspeak = ((Y_pred - Y).pow(2) * X_mask).sum() / X_mask.sum()

        if grad_enabled:
            model.zero_grad()
            losspeak.backward()
            opt.step()

        batch_losses.append(losspeak.item())

        torch.set_grad_enabled(True)

    loss = np.mean(batch_losses)

    return loss


def train():
    """Train RNN models with a decay on plateau policy

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        device (str): default cpu else cuda
        debug (Bool): if debug just run 10 epochs
    """

    # Get conf parameters
    settings = conf.get_settings()

    # device = "cpu" if not torch.cuda.is_available() else "cuda"
    device = 'cuda' if settings.use_cuda else 'cpu'

    list_data_train, list_data_val = data_loader(settings)

    list_features = [
        "FLUXCAL_g",
        "FLUXCAL_i",
        "FLUXCAL_r",
        "FLUXCAL_z",
        "FLUXCALERR_g",
        "FLUXCALERR_i",
        "FLUXCALERR_r",
        "FLUXCALERR_z",
        "delta_time",
    ]

    input_size = len(list_features)

    model = VanillaRNN(input_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ###############################
    # Begin training
    ###############################
    epoch_train_losses = []
    epoch_valid_losses = []

    for epoch in range(settings.nb_epoch):

        train_loss = batch_loop(
            model, opt, list_data_train, list_features, grad_enabled=True
        )
        valid_loss = batch_loop(
            model, opt, list_data_val, list_features, grad_enabled=False
        )

        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)

        print(f"{epoch} Training loss: {train_loss:.2f} -- Valid loss: {valid_loss:.2f}")

        plt.figure()
        plt.plot(epoch_train_losses, label="Train loss")
        plt.plot(epoch_valid_losses, label="Valid loss")
        plt.legend()
        plt.savefig(f"{settings.dump_dir}/models/loss_peak.png")
        plt.clf()
        plt.close("all")

        if epoch % 10 == 0:
            plot_predictions(model, list_data_train, list_features, "train", settings)
            plot_predictions(model, list_data_val, list_features, "val", settings)
            
            torch.save(
                        model.state_dict(),
                        f"{settings.dump_dir}/models/model.pt",
                    )

def plot_predictions(model, list_data, list_features, title, settings):

    # device = "cpu" if not torch.cuda.is_available() else "cuda"
    device = 'cuda' if settings.use_cuda else 'cpu'

    input_size = len(list_features)
    idxs = np.random.choice(len(list_data), 8)

    data = [list_data[i] for i in idxs]

    batch_size = len(idxs)
    max_length = max(map(len, data))

    X = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
    Y = np.zeros((batch_size, max_length, 1), dtype=np.float32)
    lengths = np.zeros((batch_size,), dtype=np.int64)

    for pos, df in enumerate(data):
        X[pos, : len(df), :] = df[list_features].values.astype(np.float32)
        Y[pos, : len(df), 0] = df["delta_PEAKMJDNORM"].values.astype(np.float32)
        lengths[pos] = len(df)

    torch.set_grad_enabled(False)

    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)
    lengths = torch.from_numpy(lengths).to(device)

    X_mask = (
        torch.arange(max_length).view(1, -1).to(device) < lengths.view(-1, 1)
    ).float()

    Y = Y.squeeze(-1).cpu().numpy()
    Y_pred = model(X, X_mask).squeeze(-1).cpu().numpy()

    for idx in range(len(idxs)):

        df = data[idx]

        plt.figure()
        gs = gridspec.GridSpec(2, 1)
        # Plot the lightcurve
        ax = plt.subplot(gs[0])
        for flt in ["r", "g", "z", "i"]:
            tmp = df[df[f"FLUXCAL_{flt}"] != 0]

            if len(tmp) > 0:
                arr_flux = tmp[f"FLUXCAL_{flt}"].values
                arr_fluxerr = tmp[f"FLUXCAL_{flt}"].values
                arr_time = tmp["time"].values

                ax.errorbar(
                    arr_time,
                    arr_flux,
                    yerr=arr_fluxerr,
                    fmt="o",
                    label=f"Filter {flt}",
                    color=FILTER_COLORS[flt],
                )
        ax.set_ylabel("FLUXCAL")

        # plot peak MJD preds
        ax = plt.subplot(gs[1])
        # truth
        ax.plot(df["time"].values, Y[idx, : len(df)], color="grey", linestyle="dotted")
        # predicted
        ax.plot(df["time"], Y_pred[idx, : len(df)], color="C0")

        plt.savefig(f"tests/dump/lightcurves/{title}_{idx}.png")


if __name__ == "__main__":

    train()
