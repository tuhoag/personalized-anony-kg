import argparse
import glob
import itertools
import logging
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import markers
from tqdm import tqdm

import anonygraph.algorithms as algo
import anonygraph.algorithms.clustering as calgo
import anonygraph.evaluation.clusters_metrics as cmetrics
import anonygraph.utils.data as dutils
import anonygraph.utils.general as utils
import anonygraph.utils.path as putils
import anonygraph.utils.runner as rutils
import anonygraph.utils.visualization as vutils
from anonygraph.constants import *

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_k_generator_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--refresh", type=rutils.str2bool)


def run_parallel(fn, args_list, num_workers):
    return Parallel(n_jobs=num_workers)(
        delayed(fn)(args) for args in tqdm(args_list)
    )


def extract_freq(sequence):
    result = {}

    for k in sequence:
        result[k] = result.get(k, 0) + 1

    return result


def extract_freq_sequence(freq_dict):
    k_values = np.array(sorted(list(freq_dict.keys())))
    freq = np.zeros_like(k_values)

    for idx, k in enumerate(k_values):
        freq[idx] = freq_dict[k]

    return k_values, freq


def extract_ratio_freq(freq, num_entities):
    # logger.debug(freq)
    # raise Exception(freq)
    result = {}

    for k, count in freq.items():
        result[k] = count / num_entities

    return result


def extract_info_from_path(path):
    path_info = putils.extract_info_from_gen_path(path)
    logger.debug("path {} - info: {}".format(path, path_info))

    path_info["k_sequence"] = dutils.load_k_values_sequence(
        path_info["data"], path_info["sample"], path_info["gen"], path_info
    )

    logger.debug("sequence: {}".format(path_info["k_sequence"]))

    freq_dict = extract_freq(path_info["k_sequence"])
    k_values, k_freq = extract_freq_sequence(freq_dict)

    # extract freq
    path_info["k_values"] = k_values
    path_info["k_freq"] = k_freq
    path_info["k_ratio_freq"] = k_freq / len(path_info["k_sequence"])

    return path_info


def visualize_differences_between_generations(data_name, sample, num_workers):
    dir_path = putils.get_k_values_dir_path(data_name, sample)
    logger.debug(dir_path)

    # find all k values
    k_values_paths = glob.glob(dir_path + "/*.txt")
    logger.debug(
        "found {} paths: {}".format(len(k_values_paths), k_values_paths)
    )

    # extract all info from these k values' filenames
    sequence_info_list = list(
        run_parallel(extract_info_from_path, k_values_paths, num_workers)
    )
    logger.debug("sequence info: {}".format(sequence_info_list))

    # generate groups that have the same gen_str
    gen_groups_dict = {}
    for sequence_info in sequence_info_list:
        group = gen_groups_dict.get(sequence_info["gen_str"], None)
        if group is None:
            group = []
            gen_groups_dict[sequence_info["gen_str"]] = group

        group.append(sequence_info)

    logger.debug("groups: {}".format(gen_groups_dict))

    # for each group, calculate the difference (min, max, mean, std)
    dif_stats = {}
    for key, group in gen_groups_dict.items():
        num_entities = len(group[0]["k_values"])
        max_k = int(group[0]["max_k_ratio"] * num_entities)
        min_k = int(group[0]["min_k_ratio"] * num_entities)

        array = np.array(list(map(lambda info: info["k_values"], group)))

        logger.debug("key: {} - array: {}".format(key, array.shape))

        logger.debug("sequence array: {}".format(array))

        min_sequence = np.min(array, axis=0)
        logger.debug("min sequence shape: {}".format(min_sequence.shape))
        logger.debug("min sequence: {}".format(min_sequence))
        max_sequence = np.max(array, axis=0)
        logger.debug("max sequence: {}".format(max_sequence))

        dif_sequence = max_sequence - min_sequence
        logger.debug("dif sequence: {}".format(dif_sequence))

        dif_ratio_sequence = dif_sequence / (max_k - min_k)

        dif_stats[key] = {
            "min_dif": np.min(dif_sequence),
            "max_dif": np.max(dif_sequence),
            "mean_dif": np.mean(dif_sequence),
            "min_ratio_dif": np.min(dif_ratio_sequence),
            "max_ratio_dif": np.max(dif_ratio_sequence),
            "mean_ratio_dif": np.mean(dif_ratio_sequence),
            "max_k": max_k,
            "min_k": min_k,
        }

    logger.debug(dif_stats)
    for key, stat in dif_stats.items():
        logger.info("key: {} - stat: {}".format(key, stat))


def prepare_data_for_histogram_visualization(data_name, sample, num_workers):
    # get all paths
    n_gen = 0
    dir_path = putils.get_k_values_dir_path(data_name, sample)
    logger.debug(dir_path)

    k_values_paths = glob.glob(dir_path + "/*#{}.txt".format(n_gen))
    logger.debug(
        "found {} paths: {}".format(len(k_values_paths), k_values_paths)
    )

    # for each path, extract settings info (param, max k, min k, step k), freq, freq ratio
    # extract all info from these k values' filenames
    sequence_info_list = list(
        run_parallel(extract_info_from_path, k_values_paths, num_workers)
    )
    logger.debug("sequence info: {}".format(sequence_info_list))

    return sequence_info_list


def visualize_histogram(data, param_name, key_names, conditions):
    # filter based on conditions
    # logger.debug(data)
    # raise Exception()
    filtered_data = []
    key_name = "key_{}".format(param_name)

    for info in data:
        is_valid = True
        logger.debug("current info: {}".format(info))
        for metric_name, vals in conditions.items():
            logger.debug("{} - {}".format(metric_name, vals))
            metric_vals = info.get(metric_name)
            if metric_vals not in vals:
                is_valid = False

        if not is_valid:
            continue

        key_vals = []
        for key_name in key_names:
            key_vals.append(str(info[key_name]))

        # logger.debug(key_vals)
        info[key_name] = "#".join(key_vals)
        filtered_data.append(info)

    logger.debug(
        "filtered data length: {} from {}".format(
            len(filtered_data), len(data)
        )
    )
    # raise Exception()
    unique_k_values = set()
    bar_width = 0.25

    for idx, info in enumerate(filtered_data):
        unique_k_values.update(set(info["k_values"]))
        pos = np.arange(len(info["k_values"])) + bar_width * idx
        plt.bar(
            x=pos,
            height=info["k_ratio_freq"],
            width=bar_width,
            label=info[param_name],
            edgecolor='white'
        )

        # break

    unique_k_values_list = sorted(list(unique_k_values))
    logger.debug("unique k values: {}".format(unique_k_values_list))

    title_metrics = []
    for metric_name, vals in conditions.items():
        title_metrics.append("{}:{}".format(metric_name, ",".join(map(str,vals))))

    plt.title("Frequency with the same value of {}".format("-".join(title_metrics)))
    plt.legend(title=param_name)
    plt.ylabel("Frequency (%)")
    plt.xlabel("k")
    plt.xticks(
        [idx + bar_width for idx, r in enumerate(unique_k_values_list)], unique_k_values_list
    )
    # plt.xticks(list(all_k_values))
    # plt.grid(linestyle="--")
    plt.show()

def visualize_k_statistics(data):
    logger.debug(data)

    for item in data:
        data_name = item["data"]
        setting = item["gen_str"]
        mean_k = np.mean(item["k_sequence"])
        logger.info("data: {} - gen: {} - mean k: {:4.2f}".format(data_name, setting, mean_k))



def visualize_k_historam_over_generations(data_name, sample, num_workers):
    # load data
    data = prepare_data_for_histogram_visualization(
        data_name, sample, num_workers
    )

    logger.debug(data)
    visualize_k_statistics(data)
    # visualize_histogram(
    #     data=data,
    #     param_name="za",
    #     key_names=["min_k_ratio", "max_k_ratio", "range_k_ratio"],
    #     conditions={
    #         "gen": "zipf",
    #         "min_k_ratio": [0.002],
    #         "max_k_ratio": [0.005],
    #         "range_k_ratio": [0.001],
    #     }
    # )

    # visualize_histogram(
    #     data=data,
    #     param_name="za",
    #     key_names=["min_k_ratio", "max_k_ratio", "range_k_ratio"],
    #     conditions={
    #         "gen": "zipf",
    #         "min_k_ratio": [0.01],
    #         "max_k_ratio": [0.05],
    #         "range_k_ratio": [0.005],
    #     }
    # )

    # visualize_histogram(
    #     data=data,
    #     param_name="max_k",
    #     key_names=["min_k", "range_k"],
    #     conditions={
    #         "gen": "rzipf",
    #         "min_k": [5],
    #         # "max_k_ratio": [0.05],
    #         # "range_k": [0.005],
    #     }
    # )


def main(args):
    logger.debug(args)
    data_name = args["data"]
    sample = args["sample"]
    num_workers = args["workers"]

    # differences between generations
    # visualize_differences_between_generations(data_name, sample, num_workers)

    # distributions of k values under three params: a, max k ratio, min k ratio, step k ratio
    visualize_k_historam_over_generations(data_name, sample, num_workers)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
