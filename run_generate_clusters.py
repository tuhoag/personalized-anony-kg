import argparse
import glob
import itertools
import logging
import os

from joblib import Parallel, delayed

import anonygraph.algorithms.clustering as calgo
import anonygraph.k_generators as generators
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.runner as rutils
from anonygraph.constants import *

os.system("taskset -p 0xff %d" % os.getpid())

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_clustering_runner_argument(parser)
    rutils.add_k_generator_runner_argument(parser)
    rutils.add_info_loss_argument(parser)
    rutils.add_cluster_constraint_enforcer_runner_argument(parser)
    rutils.add_log_argument(parser)
    rutils.add_workers_argument(parser)
    parser.add_argument("--args_mode")
    rutils.add_args_list_argument(parser, "run_mode", rutils.str2bool)
    # parser.add_argument("--run_mode", type=rutils.str2list(rutils.str2bool))
    parser.add_argument("--refresh", type=rutils.str2bool, default=False)


def run_generate_raw_clusters(args_list):
    file_name = 'generate_raw_clusters.py'
    Parallel(n_jobs=args['workers'], backend="multiprocessing")(
        delayed(rutils.run_python_file)(file_name, args_item)
        for args_item in args_list
    )


def run_generate_anonymized_clusters(args_list):
    file_name = 'anonymize_clusters.py'
    Parallel(n_jobs=args['workers'], backend="multiprocessing")(
        delayed(rutils.run_python_file)(file_name, args_item)
        for args_item in args_list
    )


def get_all_clustering_args():
    clustering_args_list = [{"calgo": PSIZE_CLUSTERING_ALGORITHM}]

    calgo_list = [K_MEDOIDS_CLUSTERING_ALGORITHM, HDBSCAN_CLUSTERING_ALGORITHM]
    calgo_mode_list = ["max", "min", "mean"]

    for calgo, cargs in itertools.product(calgo_list, calgo_mode_list):
        clustering_args_list.append({
            "calgo": calgo,
            "calgo_args": cargs,
        })

    return clustering_args_list


def get_all_raw_clusters_args(args):
    args_list = []

    k_values_paths = glob.glob(
        putils.get_k_values_dir_path(args["data"], args["sample"]) + "/*"
    )
    logger.debug("found {} k values paths: {}".format(len(k_values_paths), k_values_paths))
    # raise Exception()
    clustering_args_list = get_all_clustering_args()

    for path in k_values_paths:
        current_args = putils.extract_info_from_gen_path(path)

        for clustering_args in clustering_args_list:
            new_args = current_args.copy()
            new_args.update(clustering_args)

            args_list.append(new_args)

    return args_list


def get_param_raw_clusters_args(args):
    n_gens = args["n_gens"]
    gen_name = args["gen"]
    gen_args_list = args["gen_args_list"]
    calgo_name = args["calgo"]
    calgo_args_list = args["calgo_args_list"]

    real_n_gens = generators.get_real_num_generations(gen_name, n_gens)

    args_list = []

    for gen_args, calgo_args, gen_n in itertools.product(gen_args_list, calgo_args_list, range(real_n_gens)):
        current_args = rutils.copy_args(args)

        current_args["gen_n"] = gen_n
        current_args["gen_args"] = gen_args
        current_args["calgo_args"] = calgo_args

        # logger.debug(calgo_args)

        args_list.append(current_args)

        # logger.debug(current_args)
        # raise Exception()

    return args_list


def get_param_anony_clusters_args(args):
    enforcer_args_list = args["enforcer_args_list"]

    args_list = []

    raw_clustering_args_list = get_param_raw_clusters_args(args)

    for raw_args, enforcer_args in itertools.product(raw_clustering_args_list, enforcer_args_list):
        current_args = rutils.copy_args(raw_args)

        current_args["enforcer_args"] = enforcer_args

        args_list.append(current_args)

    return args_list


def get_all_enforcers_args():
    max_dist_list = [0, 0.25, 0.5, 0.75, 1.0]

    args_list = [{
        "enforcer": SR_ENFORCER,
        "enforcer_args": None,
    }]

    for max_dist in max_dist_list:
        args_list.append({
            "enforcer": MERGE_SPLIT_ENFORCER,
            "enforcer_args": [max_dist],
        })

        args_list.append({
            "enforcer": KMEANS_PARTITION_ENFORCER,
            "enforcer_args": [max_dist],
        })

    # for max_dist, anonymity_mode in itertools.product(max_dist_list, anonymity_mode_list):
    #     args_list.append({
    #         "enforcer": KMEANS_PARTITION_ENFORCER,
    #         "enforcer_args": [max_dist, anonymity_mode],
    #     })

    return args_list


def get_all_anony_clusters_args(args):
    raw_clusters_path = glob.glob(
        putils.get_clusters_dir_path(args["data"], args["sample"], "raw") + "/*"
    )

    enforcers_args_list = get_all_enforcers_args()

    args_list = []

    all_raw_clustering_args_list = get_all_raw_clusters_args(args)

    for raw_args, enforcer_args in itertools.product(all_raw_clustering_args_list, enforcers_args_list):
        logger.debug("raw args: {} - enforcer args: {}".format(raw_args, enforcer_args))
        # if is_same_anonymity_mode(raw_args, enforcer_args):
        current_args = rutils.copy_args(raw_args)
        current_args["enforcer"] = enforcer_args["enforcer"]
        current_args["enforcer_args"] = rutils.convert_raw_val_to_str_val("enforcer_args", enforcer_args["enforcer_args"])

        args_list.append(current_args)

    return args_list

def is_same_anonymity_mode(raw_args, enforcer_args):
    logger.debug("raw_args: {} - enforcer args: {}".format(raw_args, enforcer_args))
    if enforcer_args["enforcer"] != KMEANS_PARTITION_ENFORCER or raw_args["calgo"] == PSIZE_CLUSTERING_ALGORITHM:
        return True

    enforcer_anonymity_mode = enforcer_args["enforcer_args"][1]
    raw_anonymity_mode = raw_args["calgo_args"]


    logger.debug("raw anonymity: {} - enforcer: {}".format(raw_anonymity_mode, enforcer_anonymity_mode))

    # raise Exception("raw anonymity: {} - enforcer: {}".format(raw_anonymity_mode, enforcer_anonymity_mode))
    return enforcer_anonymity_mode == raw_anonymity_mode

def remove_files_in_dir_path(dir_path, pattern="*.*"):
    count = 0
    clusters_paths = glob.glob(dir_path + "/" + pattern)
    for clusters_path in clusters_paths:
        os.remove(clusters_path)
        count += 1

    return count

def main(args):
    logger.info(args)

    anony_clusters_gen_args_list = []

    # logger.debug(args["run_mode"])
    # raise Exception()
    if args["run_mode"][0]:
        if args["refresh"]:
            # remove all raw clusters
            path = putils.get_clusters_dir_path(
                args["data"], args["sample"], "raw"
            )

            num_removed_clusters_paths = remove_files_in_dir_path(path, "*.txt")

            logger.info(
                "removed {} raw clusters in {}".format(
                    num_removed_clusters_paths, path
                )
            )

        # raise Exception()
        if args["args_mode"] == "all":
            raw_clusters_gen_args_list = get_all_raw_clusters_args(args)
        else:
            raw_clusters_gen_args_list = get_param_raw_clusters_args(args)

        logger.debug(anony_clusters_gen_args_list)

        run_generate_raw_clusters(raw_clusters_gen_args_list)

    if args["run_mode"][1]:
        if args["refresh"]:
            # remove all anony clusters
            path = putils.get_clusters_dir_path(
                args["data"], args["sample"], "anony"
            )

            num_removed_clusters_paths = remove_files_in_dir_path(path, "*.txt")

            logger.info(
                "removed {} anony clusters in {}".format(
                    num_removed_clusters_paths, path
                )
            )

        if args["args_mode"] == "all":
            anony_clusters_gen_args_list = get_all_anony_clusters_args(args)
        else:
            anony_clusters_gen_args_list = get_param_anony_clusters_args(args)

        logger.debug(anony_clusters_gen_args_list)
        run_generate_anonymized_clusters(anony_clusters_gen_args_list)




if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
