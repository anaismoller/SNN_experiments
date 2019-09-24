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
            batch_first=False,
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
            x, lengths, batch_first=False, enforce_sorted=True
        )

        x_packed, hidden = self.rnn_layer(x_packed)

        # unpack
        x_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x_packed, batch_first=False
        )
        # x_padded is (B, L, D)

        x_pred = self.output_peak_layer(x_padded)

        return x_pred


# def get_data_batch(list_data, batch_idxs, settings):

#     start, end = batch_idxs[0], batch_idxs[-1] + 1

#     data = list_data[start:end]

#     batch_size = len(batch_idxs)
#     input_size = data[0][0].shape[-1]
#     max_length = max([len(x[0]) for x in data])

#     X = np.zeros((batch_size, max_length, input_size), dtype=np.float32)
#     Y = np.zeros((batch_size, max_length, 1), dtype=np.float32)
#     lengths = np.zeros((batch_size,), dtype=np.int64)

#     for pos, (x, target_tuple, _) in enumerate(data):

#         X[pos, : len(x), :] = x.astype(np.float32)
#         Y[pos, : len(x), 0] = target_tuple[1].astype(np.float32)
#         lengths[pos] = len(x)

#     X = torch.from_numpy(X).to(DEVICE)
#     Y = torch.from_numpy(Y).to(DEVICE)
#     lengths = torch.from_numpy(lengths).to(DEVICE)

#     X_mask = (
#         torch.arange(max_length).view(1, -1).to(DEVICE) < lengths.view(-1, 1)
#     ).float()

#     return X, Y, X_mask


def get_data_batch(list_data, batch_idxs, settings):

    start, end = batch_idxs[0], batch_idxs[-1] + 1

    data = list_data[start:end]

    batch_size = len(batch_idxs)
    input_size = data[0][0].shape[-1]
    list_lengths = [len(x[0]) for x in data]

    max_length = max(list_lengths)
    idxs_sort = np.argsort(list_lengths)[::-1]  # descending length

    X = np.zeros((max_length, batch_size, input_size), dtype=np.float32)
    Y = np.zeros((max_length, batch_size, 1), dtype=np.float32)
    lengths = np.zeros((batch_size,), dtype=np.int64)

    for pos, idx in enumerate(idxs_sort):

        x, target_tuple, _ = data[idx]

        X[: len(x), pos, :] = x.astype(np.float32)
        Y[: len(x), pos, 0] = target_tuple[1].astype(np.float32)
        lengths[pos] = len(x)

    X = torch.from_numpy(X).to(DEVICE)
    Y = torch.from_numpy(Y).to(DEVICE)
    lengths = torch.from_numpy(lengths).to(DEVICE)

    X_mask = (
        torch.arange(max_length).view(1, -1).to(DEVICE) < lengths.view(-1, 1)
    ).float()

    return X, Y, X_mask.transpose(1, 0).contiguous()


def get_evaluation_metrics(settings, list_data, model, sample_size=None):
    """Compute evaluation metrics on a list of data points

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        list_data (list): contains data to evaluate
        model (torch.nn Model): pytorch model
        sample_size (int): subset of the data to use for validation. Default: ``None``

    Returns:
        d_losses (dict) maps metrics to their computed value
    """

    # Validate
    list_target_peak = []
    list_pred_peak = []
    list_mask = []

    list_target_peak2 = []
    list_pred_peak2 = []
    list_mask2 = []

    num_elem = len(list_data)
    num_batches = num_elem // min(num_elem // 2, settings.batch_size)
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    # If required, pick a subset of list batches at random
    if sample_size:
        batch_idxs = np.random.permutation(len(list_batches))
        num_batches = sample_size // min(sample_size // 2, settings.batch_size)
        batch_idxs = batch_idxs[:num_batches]
        list_batches = [list_batches[batch_idx] for batch_idx in batch_idxs]

    for batch_idxs in list_batches:
        with torch.no_grad():
            model.eval()
            X, Y, X_mask = get_data_batch(list_data, batch_idxs, settings)
            # lengths = X_mask.sum(dim=-1).long()
            # # Pack
            # x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            #     X, lengths, batch_first=False, enforce_sorted=True
            # )

            packed_tensor, X_tensor, target_tensor, idxs_rev_sort = tu.get_data_batch(
                list_data, batch_idxs, settings
            )

            assert np.all(
                np.ravel(X_tensor.detach().cpu().numpy())
                == np.ravel(X.detach().cpu().numpy())
            )
            assert np.all(
                np.ravel(Y.detach().cpu().numpy())
                == np.ravel(target_tensor[1].detach().cpu().numpy())
            )

            _, Y_pred, mask_pred = model(packed_tensor)

            assert np.all(
                np.ravel(mask_pred.detach().cpu().numpy())
                == np.ravel(X_mask.detach().cpu().numpy())
            )

            # from SNN
            packed_tensor, X_tensor, target_tensor, idxs_rev_sort = tu.get_data_batch(
            list_data, batch_idxs, settings
            )
            
            outpeak = model(X_tensor,X_mask)
            np.testing.assert_array_equal(outpeak.view(-1).numpy(),Y_pred.view(-1).numpy())
            np.testing.assert_array_equal(target_tensor[1].view(-1).numpy(),Y.view(-1).numpy())
            #
            # works! which means it is not batching
            # could be the eval_step, to be proved

            # from SNN
            # ni with reverse sort ni without it I get the same answer
            # the problem is how tensor is aranged 
            # thibs Y = [a,b,c],[d,e,f]
            # mine is [a,d,][b,c]
            target_tensor_class, target_tensor_peak = target_tensor
            pred_peak_tensor = outpeak
            # reshape (B,L) 
            pred_peak_reshaped = pred_peak_tensor.view(-1)
            target_peak_reshaped = target_tensor_peak.view(-1)
            peak_mask_reshaped = X_mask.view(-1)
            # Flatten & convert to numpy array
            pred_peak_numpy = pred_peak_reshaped.data.cpu().numpy()
            target_peak_numpy = target_peak_reshaped.data.cpu().numpy()
            peak_mask_numpy = peak_mask_reshaped.data.cpu().numpy()

            list_pred_peak2.append(pred_peak_numpy)
            list_target_peak2.append(target_peak_numpy)
            list_mask2.append(peak_mask_numpy)
            
            list_target_peak.append(Y.view(-1).cpu().numpy())
            list_pred_peak.append(Y_pred.view(-1).cpu().numpy())
            list_mask.append(X_mask.view(-1).cpu().numpy())

    preds_peak = np.concatenate(list_pred_peak)
    targets_peak = np.concatenate(list_target_peak)
    mask = np.concatenate(list_mask)

    preds_peak2 = np.concatenate(list_pred_peak2)
    targets_peak2 = np.concatenate(list_target_peak2)
    mask2 = np.concatenate(list_mask2)

    # regression metrics
    MSE = (np.power((preds_peak - targets_peak), 2)*mask).sum() / mask.sum()
    MSE2 = (np.power((preds_peak2 - targets_peak2), 2)*mask2).sum() / mask2.sum()

    return MSE2


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
    rnn = tu.get_model(settings, len(settings.training_features))
    # rnn = VanillaRNN(len(settings.training_features))
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

        rnn.train()

        for batch_idxs in list_batches:

            X, Y, X_mask = get_data_batch(list_data_train, batch_idxs, settings)
            # lengths = X_mask.sum(dim=-1).long()
            # # Pack
            # x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            #     X, lengths, batch_first=False, enforce_sorted=True
            # )

            packed_tensor, X_tensor, target_tensor, idxs_rev_sort = tu.get_data_batch(
                list_data_train, batch_idxs, settings
            )

            _, Y_pred, mask_pred = rnn(packed_tensor)

            assert np.all(
                np.ravel(X_tensor.detach().cpu().numpy())
                == np.ravel(X.detach().cpu().numpy())
            )
            assert np.all(
                np.ravel(Y.detach().cpu().numpy())
                == np.ravel(target_tensor[1].detach().cpu().numpy())
            )

            assert np.all(
                np.ravel(mask_pred.detach().cpu().numpy())
                == np.ravel(X_mask.detach().cpu().numpy())
            )

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

                X, Y, X_mask = get_data_batch(list_data_val, batch_idxs, settings)
                # lengths = X_mask.sum(dim=-1).long()
                # # Pack
                # x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                #     X, lengths, batch_first=False, enforce_sorted=True
                # )

                packed_tensor, X_tensor, target_tensor, idxs_rev_sort = tu.get_data_batch(
                    list_data_val, batch_idxs, settings
                )

                _, Y_pred, mask_pred = rnn(packed_tensor)

                assert np.all(
                    np.ravel(X_tensor.detach().cpu().numpy())
                    == np.ravel(X.detach().cpu().numpy())
                )
                assert np.all(
                    np.ravel(Y.detach().cpu().numpy())
                    == np.ravel(target_tensor[1].detach().cpu().numpy())
                )

                assert np.all(
                    np.ravel(mask_pred.detach().cpu().numpy())
                    == np.ravel(X_mask.detach().cpu().numpy())
                )

            # Loss
            Y = Y.view(-1)
            Y_pred = Y_pred.view(-1)
            X_mask = X_mask.view(-1)
            # Y, Y_pred, X_mask all are (B * L)
            losspeak = ((Y_pred - Y).pow(2) * X_mask).sum() / X_mask.sum()

            batch_losses.append(losspeak.item())

        loss = np.mean(batch_losses)
        d_monitor_val["reg_MSE"].append(loss)

        # Get metrics (subsample training set to same size as validation set for speed)
        mse_train = get_evaluation_metrics(
            settings, list_data_train, rnn, sample_size=len(list_data_val)
        )
        mse_val = get_evaluation_metrics(settings, list_data_val, rnn, sample_size=None)

        d_train = tu.get_evaluation_metrics(
            settings, list_data_train, rnn, sample_size=len(list_data_val)
        )["reg_MSE"]
        d_val = tu.get_evaluation_metrics(
            settings, list_data_val, rnn, sample_size=None
        )["reg_MSE"]

        print()
        print()
        print()
        print("Train", d_monitor_train["reg_MSE"][-1])
        print("Train numpu", mse_train)
        print("Train other", d_train)
        print()
        print("Val", d_monitor_val["reg_MSE"][-1])
        print("Val numpu", mse_val)
        print("Val other", d_val)


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
