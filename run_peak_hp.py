import os
import json
import torch
import shlex
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from itertools import product
from supernnova.paper.superNNova_plots import plot_speed_benchmark
from supernnova.utils import logging_utils as lu

"""superNNova paper experiments
"""

LIST_SEED = [0, 100, 1000, 55, 30496]


def run_cmd(cmd, debug, seed):
    """Run command
    Using cuda if available
    """

    cmd += f" --seed {seed} "

    if torch.cuda.is_available():
        cmd += " --use_cuda "

    if debug is True:
        # Run for 1 epoch only
        cmd += "--cyclic_phases 1 1 1 "
        cmd += "--nb_epoch 1 "

        if "num_inference_samples" not in cmd:
            # Make inference faster
            cmd = cmd + "--num_inference_samples 2 "

    subprocess.check_call(shlex.split(cmd))


def run_data(dump_dir, raw_dir,fits_dir, debug, seed):
    """Create database
    """

    cmd = "python -W ignore run.py --data " f"--dump_dir {dump_dir} --raw_dir {raw_dir} --fits_dir {fits_dir}"
    run_cmd(cmd, debug, seed)


def run_baseline_hp(dump_dir, debug, seed):

    lu.print_green(f"SEED {seed}: BASELINE HP")

    if seed != LIST_SEED[0]:
        return

    list_batch_size = [64, 128, 512]
    list_num_layers = [1, 2, 3]
    list_layer_type = ["gru", "lstm"]
    list_bidirectional = [True, False]
    list_rnn_output_option = ["standard", "mean"]
    list_random_length = [True, False]
    list_hidden_dim = [16, 32]
    list_peak_norm = [None, 'basic','log']

    if debug is True:
        list_batch_size = list_batch_size[:1]
        list_hidden_dim = list_hidden_dim[:1]

    for (
        batch_size,
        num_layers,
        layer_type,
        bidirectional,
        rnn_output_option,
        random_length,
        hidden_dim, 
        peak_norm,
    ) in product(
        list_batch_size,
        list_num_layers,
        list_layer_type,
        list_bidirectional,
        list_rnn_output_option,
        list_random_length,
        list_hidden_dim,
        list_peak_norm
    ):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--dump_dir {dump_dir} "
            f"--cyclic "
            f"--data_fraction 0.2 "
            f"--batch_size {batch_size} "
            f"--layer_type {layer_type} "
            f"--num_layers {num_layers} "
            f"--bidirectional {bidirectional} "
            f"--random_length {random_length} "
            f"--rnn_output_option {rnn_output_option} "
            f"--hidden_dim {hidden_dim} "
        )
        if peak_norm:
            cmd += f"--peak_norm {peak_norm} "
        run_cmd(cmd, debug, seed)


def run_baseline_tmp(dump_dir, debug, seed):

    lu.print_green(f"SEED {seed}: BASELINE HP")

    if seed != LIST_SEED[0]:
        return

    list_peak_norm = [None, 'basic','log']
    list_random_start = [False,True]

    for (
        peak_norm,
        random_start,
    ) in product(
        list_peak_norm,
        list_random_start,
    ):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--dump_dir {dump_dir} "
            f"--cyclic "
        )
        if peak_norm:
            cmd += f"--peak_norm {peak_norm} "
        if random_start:
            cmd += f"--random_start"
        run_cmd(cmd, debug, seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SNIa classification")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    default_dump_dir = Path(dir_path).parent / "snn_peak_dump"

    parser.add_argument(
        "--dump_dir",
        type=str,
        default=default_dump_dir,
        help="Default path where models are dumped",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=default_dump_dir,
        help="Default path where raw data is",
    )
    parser.add_argument(
        "--fits_dir",
        type=str,
        default=default_dump_dir,
        help="Default path where fits are",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Switch to debug mode: will run dummy experiments to quickly check the whole pipeline",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=LIST_SEED,
        nargs="+",
        choices=LIST_SEED,
        help="Seed with which to run the experiments",
    )
    args = parser.parse_args()

    list_seeds = args.seeds[:2] if args.debug else args.seeds

    for seed in list_seeds:

        if seed == list_seeds[0]:
            ############################
            # Data
            ############################
            # run_data(args.dump_dir, args.raw_dir, args.fits_dir, args.debug, seed)

            ##################
            # Hyperparams
            ##################
            # run_baseline_hp(args.dump_dir, args.debug, seed)
            run_baseline_tmp(args.dump_dir, args.debug, seed)
