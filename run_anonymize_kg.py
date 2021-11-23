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
    # rutils.add_clustering_runner_argument(parser)
    # rutils.add_k_generator_runner_argument(parser)
    # rutils.add_info_loss_argument(parser)
    # rutils.add_cluster_constraint_enforcer_runner_argument(parser)
    rutils.add_log_argument(parser)
    rutils.add_workers_argument(parser)
    parser.add_argument("--args_modes", type=rutils.str2list(str))
    parser.add_argument("--max_dist_list", type=rutils.str2list(float))
    parser.add_argument("--gen_list", type=rutils.str2list(str,","))
    parser.add_argument("--gen_args_list", type=rutils.str2list(str,"-"))
    parser.add_argument("--calgo_list", type=rutils.str2list(str, ","))
    parser.add_argument("--calgo_args_list", type=rutils.str2list(str, ","))
    # rutils.add_args_list_argument(parser, "run_mode", rutils.str2bool)
    # parser.add_argument("--run_mode", type=rutils.str2list(rutils.str2bool))
    # parser.add_argument("--refresh", type=rutils.str2bool, default=False)



def run_anonymize_kgs(args_list):
    file_name = 'anonymize_kg.py'
    results = Parallel(n_jobs=args['workers'], backend="multiprocessing")(
        delayed(rutils.run_python_file)(file_name, args_item)
        for args_item in args_list
    )

    logger.info("results (len: {}): {}".format(len(results), sum(results)))



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


def get_all_enforcers_args(enforcer_name, max_dist_list=[0, 0.25, 0.5, 0.75, 1.0]):
    # max_dist_list = [0, 0.25, 0.5, 0.75, 1.0]
    # max_dist_list = [0.5, 1.0]

    args_list = []
    if enforcer_name == SR_ENFORCER:
        args_list.append({
            "enforcer": SR_ENFORCER,
            "enforcer_args": None,
        })
    elif enforcer_name == MERGE_SPLIT_ENFORCER:
        for max_dist in max_dist_list:
            args_list.append({
                "enforcer": MERGE_SPLIT_ENFORCER,
                "enforcer_args": str(max_dist),
            })
    else:
        raise Exception("Unsupported enforcer: {}".format(enforcer_name))

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

def get_all_k_gen_args_list(gen_list, range_list):
    args_list = []

    # gen_list = [RANGE_ZIPF_GENERATOR, RANGE_TOTAL_EDGES_GENERATOR]
    n_gens = 3

    for gen, range_args in zip(gen_list, range_list):
        if gen == RANGE_ZIPF_GENERATOR:
            for gen_n in range(n_gens):
                args_list.append({
                    "gen": gen,
                    "gen_n": gen_n,
                    "gen_args": range_args + ",2"
                })
        elif gen == RANGE_TOTAL_EDGES_GENERATOR:
            args_list.append({
                "gen": gen,
                "gen_args": range_args,
                "gen_n": 0,
            })

    return args_list

def get_all_clustering_args_list(calgo_list, cargs_list):


    args_list = []
    for calgo, cargs in zip(calgo_list, cargs_list):
        if calgo == "vac":
            args_list.append({
                "calgo": calgo,
                "calgo_args": None,
            })
        elif calgo in ["km", "hdbscan"]:
            args_list.append({
                "calgo": calgo,
                "calgo_args": cargs,
            })
        else:
            raise Exception("Unsupported calgo: {}".format(calgo))

    return args_list


def get_vac_exp_args(args):
    gen_list = args["gen_list"]
    gen_args_list = args["gen_args_list"]
    calgo_list = args["calgo_list"]
    cargs_list = args["calgo_args_list"]

    k_gen_args_list = get_all_k_gen_args_list(gen_list, gen_args_list)

    clustering_args_list = get_all_clustering_args_list(calgo_list, cargs_list)

    args_list = []

    for k_gen_args, calgo_args in itertools.product(k_gen_args_list, clustering_args_list):
        current_args = args.copy()
        current_args.update(k_gen_args)
        current_args.update(calgo_args)
        current_args.update({
            "enforcer": SR_ENFORCER
        })
        logger.debug("{} - {}#{} - {}#{} - {}".format(current_args["gen_n"], current_args["gen"], current_args["gen_args"], current_args["calgo"], current_args["calgo_args"], current_args["enforcer"]))
        args_list.append(current_args)

    logger.debug("VAC args list: {}".format(args_list))
    return args_list


def get_ms_exp_args(args):
    gen_list = [RANGE_ZIPF_GENERATOR]
    gen_args_list = ["5,50,5"]
    calgo_list = args["calgo_list"]
    cargs_list = args["calgo_args_list"]
    max_dist_list = args["max_dist_list"]

    k_gen_args_list = get_all_k_gen_args_list(gen_list, gen_args_list)

    clustering_args_list = get_all_clustering_args_list(calgo_list, cargs_list)

    enforcer_args = get_all_enforcers_args(MERGE_SPLIT_ENFORCER, max_dist_list)
    logger.debug("enforcer_args: {}".format(enforcer_args))
    args_list = []
    for k_gen_args, calgo_args, enforcer_args in itertools.product(k_gen_args_list, clustering_args_list, enforcer_args):
        current_args = args.copy()
        current_args.update(k_gen_args)
        current_args.update(calgo_args)
        current_args.update(enforcer_args)

        logger.debug("k_gen_args: {}".format(k_gen_args))
        logger.debug("calgo_args: {}".format(calgo_args))
        logger.debug("enforcer_args: {}\n".format(enforcer_args))

        args_list.append(current_args)

    # raise Exception()
    return args_list

def main(args):
    logger.info(args)
    args_modes = args["args_modes"]


    args_list = []
    for args_mode in args_modes:
        if args_mode == "vac":
            args_list.extend(get_vac_exp_args(args))
        elif args_mode == "ms":
            args_list.extend(get_ms_exp_args(args))
        else:
            raise Exception("Unsupported args mode: {}".format(args_mode))

    logger.debug(args_list)
    logger.debug(len(args_list))

    # raise Exception()
    run_anonymize_kgs(args_list)

    # logger.debug(anony_clusters_gen_args_list)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
