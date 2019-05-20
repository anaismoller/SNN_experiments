import argparse
import pandas as pd
from supernnova.utils import logging_utils as lu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Aggregating stats for CNNs")

    parser.add_argument(
        "--cnn_dump",
        type=str,
        default=f"../cnndump/",
        help="Default path where CNN dump is",
    )
    args = parser.parse_args()

    # read summary stats
    df_stats = pd.read_csv(f"{args.cnn_dump}/stats/summary_stats.csv")

    # get hp stats
    df_hp = df_stats[(df_stats['source_data'] == 'saltfit') & (
        df_stats['model_name_noseed'].str.contains('DF_0.2'))]
    # highest accuracy complete lc
    maxacc = df_hp['all_accuracy_mean'].max()
    lu.print_green(f'HP highest accuracy all {maxacc}')
    print(df_hp[df_hp['all_accuracy_mean']== maxacc]['model_name_noseed'].values)

    # best configuration
    df_best = df_stats[(df_stats['model_name_noseed'].str.contains('DF_1.0'))]
    df_best = df_best.round(2)

    # all accuracy
    lu.print_green('best accuracy all')
    pd.set_option('max_colwidth', 400)
    print(df_best[['model_name_noseed', 'all_accuracy_mean','all_accuracy_std']])

    # accuracy by number of epochs
    lu.print_green('epochs')
    df_best = df_best.round(1)
    for nepoch in [2,4,6]:
        epoch_keys = [f'epochs{nepoch}_accuracy_mean', f'epochs{nepoch}_accuracy_std', f'epochs{nepoch}_zspe_accuracy_mean',f'epochs{nepoch}_zspe_accuracy_std']
        print(df_best[['model_name_noseed']+epoch_keys])

    # accuracy around peak
    for peak in ["-2","","+2"]:
        lu.print_green(f'peak{peak}')
        print(df_best[['model_name_noseed',
                   f'PEAKMJD{peak}_accuracy_mean', f'PEAKMJD{peak}_accuracy_std']])

    # import ipdb; ipdb.set_trace()