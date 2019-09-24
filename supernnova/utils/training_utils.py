import torch.nn as nn
import torch
from . import logging_utils as lu
from ..training import bayesian_rnn
from ..training import variational_rnn
from ..training import vanilla_rnn
import os
import h5py
import json
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def normalize_arr(arr, settings, normalize_peak = False):
    """Normalize array before input to RNN

    - Log transform
    - Mean and std dev normalization

    Args:
        arr (np.array) array to normalize
        settings (ExperimentSettings): controls experiment hyperparameters

    Returns:
        (np.array) the normalized array
    """

    if settings.norm == "none":
        return arr

    if normalize_peak:
        arr_min = settings.arr_norm[-1, 0]
        arr_mean = settings.arr_norm[-1, 1]
        arr_std = settings.arr_norm[-1, 2]
        arr_to_norm = arr

        if settings.peak_norm == 'basic':
            arr_normed = (arr_to_norm - arr_mean) / arr_std

        elif settings.peak_norm == 'log':
            arr_to_norm = np.clip(arr_to_norm, arr_min, np.inf)
            arr_normed = np.log(arr_to_norm - arr_min + 1e-5)
            arr_normed = (arr_normed - arr_mean) / arr_std
        arr = arr_normed

    else:
        arr_min = settings.arr_norm[:-1, 0]
        arr_mean = settings.arr_norm[:-1, 1]
        arr_std = settings.arr_norm[:-1, 2]

        arr_to_norm = arr[:, settings.idx_features_to_normalize]

        # clipping
        arr_to_norm = np.clip(arr_to_norm, arr_min, np.inf)
        arr_normed = np.log(arr_to_norm - arr_min + 1e-5)
        arr_normed = (arr_normed - arr_mean) / arr_std
        arr[:, settings.idx_features_to_normalize] = arr_normed

    return arr


def unnormalize_arr(arr, settings, normalize_peak = False):
    """UnNormalize array

    Args:
        arr (np.array) array to normalize
        settings (ExperimentSettings): controls experiment hyperparameters

    Returns:
        (np.array) the normalized array
    """

    if settings.norm == "none":
        return arr

    if normalize_peak:
        arr_min = settings.arr_norm[-1, 0]
        arr_mean = settings.arr_norm[-1, 1]
        arr_std = settings.arr_norm[-1, 2]
        arr_to_unnorm = arr
        if settings.peak_norm == 'basic':
            arr_unnormed = (arr_to_unnorm * arr_std)+arr_mean
        elif settings.peak_norm == 'log':
            arr_to_unnorm = arr_to_unnorm * arr_std + arr_mean
            arr_unnormed = np.exp(arr_to_unnorm) + arr_min - 1E-5
        else:
            arr_unnormed = arr_to_unnorm
        arr = arr_unnormed
    else:
        arr_min = settings.arr_norm[:-1, 0]
        arr_mean = settings.arr_norm[:-1, 1]
        arr_std = settings.arr_norm[:-1, 2]
        arr_to_unnorm = arr[:, settings.idx_features_to_normalize]
        arr_to_unnorm = arr_to_unnorm * arr_std + arr_mean
        arr_unnormed = np.exp(arr_to_unnorm) + arr_min - 1E-5
        arr[:, settings.idx_features_to_normalize] = arr_unnormed

    return arr


def fill_data_list(
    idxs, arr_data, arr_target, arr_SNID, settings, n_features, desc, test=False
):
    """Utility to create a list of data tuples used as inputs to RNN model

    The ``settings`` object specifies which feature are selected

    Args:
        idxs (np.array or list): idx of data point to select
        arr_data (np.array): features
        arr_target (np.array): target
        arr_SNID (np.array): lightcurve unique ID
        settings (ExperimentSettings): controls experiment hyperparameters
        n_features (int): total number of features in arr_data
        desc (str): message to display while loading
        test (bool): If True: add more data to the list, as it is required at test time.
            Default: ``False``

    Returns:
        (list) the list of data tuples
    """

    list_data = []

    if desc == "":
        iterator = idxs
    else:
        iterator = tqdm(idxs, desc=desc, ncols=100)

    for i in iterator:

        X_all = arr_data[i].reshape(-1, n_features)

        # classification target
        target_class = int(arr_target[0][i])

        # new target with delta peak for each time step
        arr_time = np.cumsum(X_all[:, settings.idx_delta_time])
        peak_mjd = arr_target[1][i]
        target_lc_peak = peak_mjd - arr_time

        target = (target_class, target_lc_peak)
        lc = int(arr_SNID[i])

        # Keep an unnormalized copy of the data (for test and display)
        X_ori = X_all.copy()[:, settings.idx_features]
        X_normed = X_all.copy()[:, settings.idx_features]

        if test is True:
            list_data.append((X_normed, target, lc, X_all, X_ori))
        else:
            list_data.append((X_normed, target, lc))
    return list_data


def load_HDF5(settings, test=False):
    """Load data from HDF5

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        test (bool): If True: load data for test. Default: ``False``

    Returns:
        list_data_test (list) test data tuples if test is True

        or

        Tuple containing
            - list_data_train (list): training data tuples
            - list_data_val (list): validation data tuples
    """
    file_name = f"{settings.processed_dir}/database.h5"
    lu.print_green(f"Loading {file_name}")

    with h5py.File(file_name, "r") as hf:

        list_data_train = []
        list_data_val = []

        config_name = f"{settings.source_data}_{settings.nb_classes}classes"

        dataset_split_key = f"dataset_{config_name}"
        target_key = f"target_{settings.nb_classes}classes"

        n_samples = hf["data"].shape[0]
        subset_n_samples = int(n_samples * settings.data_fraction)

        idxs = np.random.permutation(n_samples)[:subset_n_samples]
        idxs = idxs[:]
        idxs_train = idxs[: subset_n_samples // 2]
        idxs_val = idxs[subset_n_samples // 2 :]
        idxs_test = idxs[subset_n_samples // 2 :]


        # idxs_train = np.where(hf[dataset_split_key][:] == 0)[0]
        # idxs_val = np.where(hf[dataset_split_key][:] == 1)[0]
        # idxs_test = np.where(hf[dataset_split_key][:] == 2)[0]

        # # Shuffle for good measure
        # np.random.shuffle(idxs_train)
        # np.random.shuffle(idxs_val)
        # np.random.shuffle(idxs_test)

        # idxs_train = idxs_train[: int(
        #     settings.data_fraction * len(idxs_train))]

        n_features = hf["data"].attrs["n_features"]

        training_features = " ".join(hf["features"][:][settings.idx_features])
        lu.print_green("Features used", training_features)

        arr_data = hf["data"][:]
        arr_target = hf['target_2classes'][:], hf["PEAKMJDNORM"][:]
        arr_SNID = hf["SNID"][:]

        list_data_train = fill_data_list(
            idxs_train,
            arr_data,
            arr_target,
            arr_SNID,
            settings,
            n_features,
            "Loading Training Set",
        )
        list_data_val = fill_data_list(
            idxs_val,
            arr_data,
            arr_target,
            arr_SNID,
            settings,
            n_features,
            "Loading Validation Set",
        )

        return list_data_train, list_data_val


def get_model(settings, input_size):
    """Create RNN model

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        input_size (int): dimension of the input data

    Returns:
        (torch.nn Model) pytorch model
    """

    if settings.model == "vanilla":
        rnn = vanilla_rnn.VanillaRNN
    elif settings.model == "variational":
        rnn = variational_rnn.VariationalRNN
    elif settings.model == "bayesian":
        rnn = bayesian_rnn.BayesianRNN

    rnn = rnn(input_size, settings)

    print(rnn)

    return rnn


def get_optimizer(settings, model):
    """Create gradient descent optimizer

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        model (torch.nn Model): the pytorch model

    Returns:
        (torch.optim) the gradient descent optimizer
    """

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    return optimizer


def get_data_batch(list_data, idxs, settings, max_lengths=None, OOD=None):
    """Create a batch in a deterministic way

    Args:
        list_data: (list) tuples of (X, target, lightcurve_ID)
        idxs: (array / list) indices of batch element in list_data
        settings (ExperimentSettings): controls experiment hyperparameters
        max_length (int): Maximum light curve length to be used Default: ``None``.
        OOD (str): Whether to modify data to create out of distribution data to be used Default: ``None``.

    Returns:
        Tuple containing
            - packed_tensor (torch PackedSequence): the packed features
            - X_tensor (torch Tensor): the features
            - target_tensor (torch Tensor): the target
    """

    list_len = []
    list_batch = []

    for pos, i in enumerate(idxs):
        X, target, *_ = list_data[i]
        # X is (L, D)
        # if OOD is not None:
        #     # Make a copy to be sure we do not alter the original data
        #     X = X.copy()
        # if OOD == "reverse":
        #     # For OOD test, reverse the sequence
        #     X = np.ascontiguousarray(X[::-1])
        # elif OOD == "shuffle":
        #     # For OOD test, shuffle X
        #     p = np.random.permutation(X.shape[0])
        #     X = X[p]
        # elif OOD == "sin":
        #     # For OOD test, set sine values to fluxes
        #     arr_flux = X[:, settings.idx_flux]
        #     arr_fluxerr = X[:, settings.idx_fluxerr]

        #     X_unnorm = unnormalize_arr(X.copy(), settings)
        #     arr_delta_time = X_unnorm[:, settings.idx_delta_time]
        #     arr_MJD = np.cumsum(arr_delta_time, axis=0)

        #     # Sine oscillations with 30 day period
        #     X[:, settings.idx_flux] = np.sin(arr_MJD * 2 * np.pi / 30) * np.max(
        #         arr_flux, axis=0, keepdims=True
        #     )
        #     X[:, settings.idx_fluxerr] = np.random.uniform(
        #         arr_fluxerr.min(), arr_fluxerr.max(), size=arr_fluxerr.shape
        #     )
        # elif OOD == "random":
        #     # For OOD test, set random fluxes and errors
        #     arr_flux = X[:, settings.idx_flux]
        #     arr_fluxerr = X[:, settings.idx_fluxerr]

        #     X[:, settings.idx_flux] = np.random.uniform(
        #         arr_flux.min(), arr_flux.max(), size=arr_flux.shape
        #     )
        #     X[:, settings.idx_fluxerr] = np.random.uniform(
        #         arr_fluxerr.min(), arr_fluxerr.max(), size=arr_fluxerr.shape
        #     )

        # if max_lengths is not None:
        #     assert settings.random_length is False
        #     assert settings.random_redshift is False
        #     X = X[: max_lengths[pos]]
        #     target = (target[0],target[1][:max_lengths[pos]])
        # if settings.random_length:
        #     # random length of lc
        #     random_length = np.random.randint(1, X.shape[0] + 1)
        #     X = X[:random_length]
        #     target = (target[0],target[1][:random_length])
        # if settings.random_start:
        #     # random start of light-curve to avoid biasing the peak prediction
        #     # at least 3 epochs left
        #     if X.shape[0] > 3:
        #         random_start = np.random.randint(0, X.shape[0]-3)
        #         X = X[random_start:]
        #         target = (target[0],target[1][random_start:])
        # if settings.redshift == "zspe" and settings.random_redshift:
        #     if np.random.binomial(1, 0.5) == 0:
        #         X[:, settings.idx_specz] = -1
        input_dim = X.shape[1]
        list_len.append(X.shape[0])
        list_batch.append((X, target))

    # Get indices to sort the batch by sequence size (needed to use packed sequences in pytorch)
    # Sequences should be arranged in decreasing length
    idx_sort = np.argsort(list_len)[::-1]
    idxs_rev_sort = np.argsort(idx_sort)  # these indices revert the sort
    max_len = list_len[idx_sort[0]]
    X_tensor = torch.zeros((max_len, len(idxs), input_dim))
    target_peak_tensor = torch.zeros((max_len, len(idxs), 1))
    list_target_class = []
    lengths = []
    # Assign values for the tensor
    for i, idx in enumerate(idx_sort):
        X, target = list_batch[idx]
        try:
            X_tensor[: X.shape[0], i, :] = torch.FloatTensor(X)
        except Exception:
            X_tensor[: X.shape[0], i, :] = torch.FloatTensor(
                torch.from_numpy(np.flip(X, axis=0).copy()))
        # processing targets independently
        target_peak_tensor[: X.shape[0], i, 0] = torch.FloatTensor(target[1])
        list_target_class.append(target[0])
        lengths.append(list_len[idx])

    # Move data to GPU if required
    if settings.use_cuda:
        X_tensor = X_tensor.cuda()
        target_tensor_peak = target_peak_tensor.cuda()
        target_tensor_class = torch.LongTensor(list_target_class).cuda()

    else:
        X_tensor = X_tensor
        target_tensor_class = torch.LongTensor(list_target_class)
        target_tensor_peak = target_peak_tensor

    # target tuple
    target_tensor = target_tensor_class,target_tensor_peak

    # Create a packed sequence
    packed_tensor = nn.utils.rnn.pack_padded_sequence(X_tensor, lengths)

    return packed_tensor, X_tensor, target_tensor, idxs_rev_sort


def train_step(
    settings,
    rnn,
    packed_tensor,
    target_tuple,
    criterion_class,
    optimizer,
    batch_size,
    num_batches,
):
    """Full training step : Forward and Backward pass

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        rnn (torch.nn Model): pytorch model to train
        packed_tensor (torch PackedSequence): input tensor in packed form
        target_tensor (torch Tensor): target tensor
        criterion_class (torch loss function): loss function to optimize
        optimizer (torch optim): the gradient descent optimizer
        batch_size (int): batch size
        num_batches (int): number of minibatches to scale KL cost in Bayesian
    """

    # Set NN to train mode (deals with dropout and batchnorm)
    rnn.train()

    target_class, target_peak = target_tuple

    # Zero out the gradients
    optimizer.zero_grad()

    # Forward pass
    outclass, outpeak, mask = rnn(packed_tensor)
    lossclass = criterion_class(outclass.squeeze(), target_class)

    # reshape the outputs to (B,L)
    outpeak = outpeak.squeeze(-1).transpose(1,0)
    target_peak = target_peak.squeeze(-1).transpose(1,0)

    # TEMPORARY
    # tmp mask only using last element
    tmp = torch.zeros(mask.shape)
    # find length of last element in mask
    max_lengths = (mask==1).sum(dim=1) - 1
    for i in range(tmp.size(0)):
        tmp[i][int(max_lengths[i])]=1
    mask = tmp

    if settings.use_cuda:
        outpeak = outpeak.cuda()
        target_peak = target_peak.cuda()
        mask = mask.cuda()

    # compute masked MSE
    losspeak = ((outpeak-target_peak).pow(2)*mask).sum()/mask.sum()

    # Special case for BayesianRNN, need to use KL loss
    if isinstance(rnn, bayesian_rnn.BayesianRNN):
        lossclass = lossclass + rnn.kl / (num_batches * batch_size)
    else: # TO DO, this I think can be deprecated
        lossclass = criterion_class(outclass.squeeze(), target_class)

    loss = lossclass + losspeak
    # loss = losspeak

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss


def eval_step(rnn, packed_tensor, batch_size):
    """Eval step: Forward pass only

    Args:
        rnn (torch.nn Model): pytorch model to train
        packed_tensor (torch PackedSequence): input tensor in packed form
        batch_size (int): batch size

    Returns:
        output (torch Tensor): output of rnn
    """

    # Set NN to eval mode (deals with dropout and batchnorm)
    rnn.eval()

    # Forward pass
    output = rnn(packed_tensor)

    return output


def plot_loss(d_train, d_val, epoch, settings):
    """Plot loss curves

    Plot training and validation logloss

    Args:
        d_train (dict of arrays): training log losses
        d_val (dict of arrays): validation log losses
        epoch (int): current epoch
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    for key in [k for k in d_train.keys() if k!='epoch']:

        plt.figure()
        plt.plot(d_train["epoch"], d_train[key],
                 label="Train %s" % key.title())
        plt.plot(d_val["epoch"], d_val[key], label="Val %s" % key.title())
        plt.legend(loc="best", fontsize=18)
        plt.xlabel("Step", fontsize=22)
        plt.tight_layout()
        plt.savefig(
            Path(settings.models_dir)
            / f"{settings.pytorch_model_name}"
            / f"train_and_val_{key}_{settings.pytorch_model_name}.png"
        )
        plt.close()
        plt.clf()




def get_loss_string(d_losses_train, d_losses_val):
    """Obtain a loss string to display training progress

    Args:
        d_losses_train (dict): maps {metric:value} for the training data
        d_losses_val (dict): maps {metric:value} for the validation data

    Returns:
        loss_str (str): the loss string to display
    """

    loss_str = "/".join(d_losses_train.keys())

    loss_str += " [T]: " + "/".join(
        [
            f"{value:.3g}" if (value is not None and key != "epoch") else "NA"
            for (key, value) in d_losses_train.items()
        ]
    )
    loss_str += " [V]: " + "/".join(
        [
            f"{value:.3g}" if (value is not None and key != "epoch") else "NA"
            for (key, value) in d_losses_val.items()
        ]
    )

    return loss_str


def save_training_results(settings, d_monitor, training_time):
    """Obtain a loss string to display training progress

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        d_monitor (dict): maps {metric:value}
        training_time (float): amount of time training took

    Returns:
        loss_str (str): the loss string to display
    """

    d_results = {"training_time": training_time}
    for key in ["AUC", "Acc"]:
        if key == "AUC" and settings.nb_classes > 2:
            d_results[key] = -1
        else:
            d_results[key] = max(d_monitor[key])
    d_results["loss"] = min(d_monitor["loss"])

    try:
        with open(Path(settings.rnn_dir) / "training_log.json", "r") as f:
            d_out = json.load(f)
    except Exception:
        d_out = {}

    with open(Path(settings.rnn_dir) / "training_log.json", "w") as f:
        d_out.update({settings.pytorch_model_name: d_results})
        json.dump(d_out, f)


#######################
# RandomForest Utils
#######################


def save_randomforest_model(settings, clf):
    """Save RandomForest model

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        clf (RandomForestClassifier): RandomForest model
    """

    filename = f"{settings.rf_dir}/{settings.randomforest_model_name}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(clf, f)
    lu.print_green("Saved model")


def load_randomforest_model(settings, model_file=None):
    """Load RandomForest model

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        model_file (str): path to saved randomforest model. Default: ``None``

    Returns:
        (RandomForestClassifier) RandomForest model
    """

    if model_file is None:
        model_file = f"{settings.rf_dir}/{settings.randomforest_model_name}.pickle"
    assert os.path.isfile(model_file)
    with open(model_file, "rb") as f:
        clf = pickle.load(f)
    lu.print_green("Loaded model")

    return clf


def train_and_evaluate_randomforest_model(clf, X_train, y_train, X_val, y_val):
    """Train a RandomForestClassifier and evaluate AUC, precision, accuracy
    on a validation set

    Args:
        clf (RandomForestClassifier): RandomForest model to fit and evaluate
        X_train (np.array): the training features
        y_train (np.array): the training target
        X_val (np.array): the validation features
        y_val (np.array): the validation target
    """
    lu.print_green("Fitting RandomForest...")
    clf = clf.fit(X_train, y_train)
    lu.print_green("Fitting complete")

    # Evaluate our classifier
    probas_ = clf.predict_proba(X_val)
    # Compute AUC and precision
    fpr, tpr, thresholds = metrics.roc_curve(y_val, probas_[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    pscore = metrics.precision_score(
        y_val, clf.predict(X_val), average="binary")
    lu.print_green("Validation AUC", roc_auc)
    lu.print_green("Validation precision score", pscore)

    lu.print_green(
        "Train data accuracy",
        100 * (sum(clf.predict(X_train) == y_train)) / X_train.shape[0],
    )
    lu.print_green(
        "Val data accuracy", 100 *
        (sum(clf.predict(X_val) == y_val)) / X_val.shape[0]
    )

    return clf


class StopOnPlateau(object):
    """
    Detect plateau on accuracy (or any metric)
    If chosen, will reduce learning rate of optimizer once in the Plateau

    .. code: python

          plateau_accuracy = tu.StopOnPlateau()
          for epoch in range(10):
              ... get metric ...
              plateau = plateau_accuracy.step(metric_value)
              if plateau is True:
                   break

    Args:
        patience (int): number of epochs to wait, after which we decrease the LR
            if the validation loss is plateauing
        reduce_lr-on_plateau (bool): If True, reduce LR after loss has not improved
            in the last patience epochs
        max_learning_rate_reduction (float): max factor by which to reduce the learning rate
    """

    def __init__(
        self, patience=10, reduce_lr_on_plateau=False, max_learning_rate_reduction=3
    ):

        self.patience = patience
        self.best = 0.0
        self.num_bad_epochs = 0
        self.is_better = None
        self.last_epoch = -1
        self.list_metric = []
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.max_learning_rate_reduction = max_learning_rate_reduction
        self.learning_rate_reduction = 0

    def step(self, metric_value, optimizer=None, epoch=None):
        current = metric_value
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Are we under .05 std in accuracy on the last 10 epochs
        self.list_metric.append(current)
        if len(self.list_metric) > 10:
            self.list_metric = self.list_metric[-10:]
            # are we in a plateau?
            # accuracy is not in percentage, so two decimal numbers is actually 4 in this notation
            if np.array(self.list_metric).std() < 0.0005:
                print("Has reached a learning plateau with", current, "\n")
                if optimizer is not None and self.reduce_lr_on_plateau is True:
                    print(
                        "Reducing learning rate by factor of ten",
                        self.learning_rate_reduction,
                        "\n",
                    )
                    for param in optimizer.param_groups:
                        param["lr"] = param["lr"] / 10.0
                    self.learning_rate_reduction += 1
                    if self.learning_rate_reduction == self.max_learning_rate_reduction:
                        return True
                else:
                    return True
            else:
                return False
