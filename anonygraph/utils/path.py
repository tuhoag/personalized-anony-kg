import ntpath
import os
import logging
import anonygraph.settings as settings
from anonygraph.constants import *

import anonygraph.utils.runner as rutils
import anonygraph.utils.path as putils

logger = logging.getLogger(__name__)

def get_raw_data_path(data_name):
    return os.path.join(settings.RAW_DATA_PATH, data_name)

def get_raw_graph_dir_str(data_name, sample):
    return "{}_{}".format(data_name, sample)

def get_output_path(data_name, sample):
    return os.path.join(
        settings.OUTPUT_DATA_PATH, get_raw_graph_dir_str(data_name, sample)
    )

def get_raw_graph_path(data_name, sample):
    output_dir = os.path.join(get_output_path(data_name, sample), "raw")
    return output_dir

def get_entity_index_path(data_name, sample):
    return os.path.join(get_raw_graph_path(data_name, sample), "entities.idx")

def get_distance_matrix_dir_path(data_name, sample):
    output_path = get_output_path(data_name, sample)
    pair_dist_path = os.path.join(output_path, "dist_matrix")
    return pair_dist_path

def get_distance_matrix_path(data_name, sample, info_loss_name, args):
    dir_path = get_distance_matrix_dir_path(data_name, sample)
    info_loss_str = get_info_loss_full_string(info_loss_name, args)

    return os.path.join(dir_path, info_loss_str + ".npy")

def get_pair_distance_dir_path(data_name, sample, info_loss_name, args):
    output_path = get_output_path(data_name, sample)
    pair_dist_path = os.path.join(output_path, "pair_dist")
    return pair_dist_path

def get_distance_pairs_path(data_name, sample, info_loss_name, data_type="tfrecord", args={}):
    dir_path = get_pair_distance_dir_path(data_name, sample, info_loss_name, args)
    info_loss_str = get_info_loss_full_string(info_loss_name, args)

    return os.path.join(dir_path, info_loss_str + ".{}".format(data_type))


def get_str_delimiter(args_name):
    delimiter_dict = {
        "calgo_args": ",",
        "info_loss_args": ",",
        "enforcer_args": ",",
        "gen_args": ",",
        "handler_args": ",",
        "calgo_args_list": "-",
        "gen_args_list": "-",
        "enforcer_args_list": "-",
        "run_mode": ","
    }

    return delimiter_dict[args_name]

def get_info_loss_full_string(info_loss_name, args):
    if info_loss_name == "adm":
        args_str = get_args_str("info_loss_args", args)

        logger.debug("args: {} - str: {}".format(args["info_loss_args"], args_str))
        name = "{}#{}".format(
            info_loss_name, args_str
        )

        # raise Exception()
    else:
        raise NotImplementedError(
            "Unsupported info loss metric: {}".format(info_loss_name)
        )

    return name

def get_k_values_dir_path(data_name, sample):
    output_path = get_output_path(data_name, sample)
    return os.path.join(output_path, "k_values")

def get_k_generator_str(k_generator_name, args):
    gen_n = args["gen_n"]
    base_str = get_k_generator_base_str(k_generator_name, args)
    gen_str = "{}#{}".format(base_str, gen_n)

    return gen_str

def get_args_str(name, args):
    # logger.debug(args)
    new_args = rutils.convert_raw_val_to_str_val(name, args[name])
    # result = get_str_delimiter(name).join(new_args)

    # if name == "info_loss_args":
    #     logger.debug("new args: {} - old: {}".format(new_args, args[name]))
    #     raise Exception()

    return new_args

def get_k_generator_base_str(k_generator_name, args):
    if k_generator_name in [SAME_K_GENERATOR, NORM_LIST_K_GENERATOR, RATIO_NORM_GENERATOR, RATIO_ZIPF_GENERATOR, RANGE_ZIPF_GENERATOR, TOTAL_DEGREE_GENERATOR, RANGE_TOTAL_EDGES_GENERATOR]:
        gen_args_str = get_args_str("gen_args", args)
        gen_str = "{}#{}".format(k_generator_name, gen_args_str)
    elif k_generator_name in [STATIC_GENERATOR]:
        gen_str = k_generator_name
    else:
        raise NotImplementedError(
            "Unsupported generator name: {}".format(k_generator_name)
        )

    return gen_str

def get_k_values_base_path(data_name, sample, k_generator_name, args):
    dir_path = get_k_values_dir_path(data_name, sample)
    gen_str = get_k_generator_base_str(k_generator_name, args)

    base_path = os.path.join(dir_path, gen_str)
    return base_path


def get_k_values_path(data_name, sample, k_generator_name, args):
    dir_path = get_k_values_dir_path(data_name, sample)
    gen_str = get_k_generator_str(k_generator_name, args)

    return os.path.join(dir_path, gen_str + ".txt")

def get_clusters_dir_path(data_name, sample, anony_mode):
    output_path = get_output_path(data_name, sample)
    return os.path.join(output_path, "clusters", anony_mode)

def get_anony_graphs_dir_path(data_name, sample):
    output_path = get_output_path(data_name, sample)
    return os.path.join(output_path, "graphs")

def get_outliers_handler_str(handler_name, args):
    if handler_name == NO_REMOVAL_HANDLER:
        return NO_REMOVAL_HANDLER
    elif handler_name == COST_ZSCORE_HANDLER:
        args_str = get_str_delimiter("handler_args").join(args["handler_args"])
        return "{}#{}".format(COST_ZSCORE_HANDLER, args_str)
    else:
        raise NotImplementedError("Not supported {} handler".format(handler_name))

def get_raw_clusters_str(k_generator_name, info_loss_name, handler_name, calgo_name, args):
    k_gen_str = get_k_generator_str(k_generator_name, args)
    ifn_str = get_info_loss_full_string(info_loss_name, args)
    calgo_str = get_clustering_str(calgo_name, args)
    handler_str = get_outliers_handler_str(handler_name, args)
    return "{gen}_{ifn}_{oh}_{calgo}".format(gen=k_gen_str, ifn=ifn_str, oh=handler_str, calgo=calgo_str)

def get_clustering_str(calgo_name, args):
    if calgo_name in [K_MEDOIDS_CLUSTERING_ALGORITHM, HDBSCAN_CLUSTERING_ALGORITHM]:
        cargs_str = get_args_str("calgo_args", args)
        return "{}#{}".format(calgo_name, cargs_str)
    elif calgo_name in [PSIZE_CLUSTERING_ALGORITHM, PERMULE_CLUSTERING_ALGORITHM]:
        return calgo_name
    else:
        raise NotImplementedError("Unsupported clustering algorithm: {}".format(calgo_name))

def get_raw_clusters_path(data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, args):
    raw_dir_path = get_clusters_dir_path(data_name, sample, "raw")
    raw_clusters_name = get_raw_clusters_str(k_generator_name, info_loss_name, handler_name, calgo_name, args)

    return os.path.join(raw_dir_path, raw_clusters_name + ".txt")

def get_enforcer_name(enforcer_name, args):
    args_str = get_args_str("enforcer_args", args)

    if enforcer_name in [MERGE_SPLIT_ENFORCER, KMEANS_PARTITION_ENFORCER]:
        return "{}#{}".format(enforcer_name, args_str)
    elif enforcer_name == SR_ENFORCER:
        return SR_ENFORCER
    else:
        raise NotImplementedError("Unsupported enforcer: {}".format(enforcer_name))


def get_anony_clusters_str(k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args):
    raw_cluster_str = get_raw_clusters_str(k_generator_name, info_loss_name, handler_name, calgo_name, args)
    enforcer_str = get_enforcer_name(enforcer_name, args)

    return "{}_{}".format(raw_cluster_str, enforcer_str)

def get_anony_clusters_path(data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args):
    anony_dir_path = get_clusters_dir_path(data_name, sample, "anony")
    anony_clusters_name = get_anony_clusters_str(k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args)

    return os.path.join(anony_dir_path, anony_clusters_name + ".txt")

def  get_anony_graph_path(data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args):
    anony_dir_path = get_anony_graphs_dir_path(data_name, sample)
    anony_clusters_name = get_anony_clusters_str(k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args)

    return os.path.join(anony_dir_path, anony_clusters_name)


def get_tuning_exp_data_path(exp_name, data_name, sample, args):
    data_str = get_raw_graph_dir_str(data_name, sample)

    return os.path.join(
        settings.EXP_DATA_PATH, exp_name,
        "{}.csv".format(data_str)
    )

def get_agg_tuning_exp_data_path(exp_name, data_name, sample, args):
    data_str = get_raw_graph_dir_str(data_name, sample)

    return os.path.join(
        settings.EXP_DATA_PATH, exp_name,
        "{}_agg.csv".format(data_str)
    )

def get_tuning_graphs_exp_data_path(exp_name, data_name, sample, args):
    data_str = get_raw_graph_dir_str(data_name, sample)

    return os.path.join(
        settings.EXP_DATA_PATH, exp_name,
        "{}.csv".format(data_str)
    )

def get_agg_tuning_graphs_exp_data_path(exp_name, data_name, sample, args):
    data_str = get_raw_graph_dir_str(data_name, sample)

    return os.path.join(
        settings.EXP_DATA_PATH, exp_name,
        "{}_agg.csv".format(data_str)
    )

def get_tuning_clusters_fig_dir_path(exp_name):
    return os.path.join(
        settings.FIG_PATH, exp_name
    )

def extract_info_from_data_str(data_str):
    try:
        splits = data_str.split("_")
        return {
            "data": splits[0],
            "sample": int(splits[1]),
        }
    except Exception as e:
        logger.error("str: {} - splits: {}".format(data_str, splits))
        raise e


def extract_info_from_anony_mode_str(anony_mode_str):
    return {
        "anony_mode": anony_mode_str
    }

def extract_info_from_outliers_handler_str(handler_str):
    splits = handler_str.split("#")
    return {
        "handler_str": handler_str
    }

def extract_info_from_clusters_str(clusters_str):
    splits = clusters_str.split("_")

    info = extract_info_from_gen_str(splits[0])
    info.update(extract_info_from_info_loss_str(splits[1]))
    info.update(extract_info_from_outliers_handler_str(splits[2]))
    info.update(extract_info_from_calgo_str(splits[3]))

    if len(splits) == 5:
        info.update(extract_info_from_enforcer_str(splits[4]))

    return info

def extract_info_from_info_loss_str(info_loss_str):
    info_loss_name, args = info_loss_str.split("#")
    args_list = list(map(float, args.split(get_str_delimiter("info_loss_args"))))

    info_loss_args_str = rutils.convert_raw_val_to_str_val("info_loss_args", args)
    # info_loss_args_str = get_args_str("info_loss_args", args)

    return {
        "info_loss": info_loss_name,
        "alpha_adm": args_list[0],
        "alpha_dm": args_list[1],
        "info_loss_args": args_list,
        "info_loss_args_str": info_loss_args_str,
    }

def extract_info_from_enforcer_str(enforcer_str):
    splits = enforcer_str.split("#")
    enforcer_name = splits[0]

    if len(splits) == 2:
        enforcer_args_str = splits[1]
        enforcer_args = enforcer_args_str.split(get_str_delimiter("enforcer_args"))
    else:
        enforcer_args = []
        enforcer_args_str = ""

    info = {
        "enforcer": enforcer_name,
        "enforcer_str": enforcer_str,
        "enforcer_args": enforcer_args_str,
    }

    if enforcer_name in [MERGE_SPLIT_ENFORCER, KMEANS_PARTITION_ENFORCER]:
        info.update({
            "max_dist": float(enforcer_args[0]),
        })
    return info


def extract_info_from_calgo_str(calgo_str):
    splits = calgo_str.split("#")

    calgo = splits[0]


    if calgo in [K_MEDOIDS_CLUSTERING_ALGORITHM, HDBSCAN_CLUSTERING_ALGORITHM]:
        return {
            "calgo": calgo,
            "calgo_args": splits[1],
            "calgo_str": calgo_str,
        }
    # elif calgo in [PSIZE_CLUSTERING_ALGORITHM]:

    return {
        "calgo": calgo_str,
        "calgo_str": calgo_str,
    }


def extract_info_from_gen_str(gen_str):
    splits = gen_str.split("#")

    logger.debug(splits)
    gen_name = splits[0]
    gen_args = splits[1]
    gen_n = int(splits[-1])

    gen_name_str = "#".join(splits[:2])

    info = {
        "gen": gen_name,
        "gen_args": gen_args,
        "gen_str": gen_name_str,
        "gen_n": gen_n,
    }

    if gen_name in [RATIO_ZIPF_GENERATOR]:
        args_splits = gen_args.split(",")

        info.update({
            "max_k_ratio": float(args_splits[1]),
            "min_k_ratio": float(args_splits[0]),
            "range_k_ratio": float(args_splits[2]),
            "za": float(args_splits[3]),
        })
    elif gen_name in [TOTAL_DEGREE_GENERATOR]:
        args_splits = gen_args.split(",")

        info.update({
            "max_k_ratio": float(args_splits[1]),
            "min_k_ratio": float(args_splits[0]),
            "range_k_ratio": float(args_splits[2]),
        })
    elif gen_name in [RANGE_ZIPF_GENERATOR]:
        args_splits = gen_args.split(",")

        info.update({
            "max_k": int(args_splits[1]),
            "min_k": int(args_splits[0]),
            "range_k": int(args_splits[2]),
            "za": float(args_splits[3]),
        })
    elif gen_name in [RANGE_TOTAL_EDGES_GENERATOR]:
        args_splits = gen_args.split(",")

        info.update({
            "max_k": int(args_splits[1]),
            "min_k": int(args_splits[0]),
            "range_k": int(args_splits[2]),
        })

    return info

def extract_info_from_gen_path(path):
    splits = path.split(".txt")[0].split(os.path.sep)

    # raise Exception(splits)
    logger.debug(splits)

    info = extract_info_from_data_str(splits[-3])
    info.update(extract_info_from_gen_str(splits[-1]))

    return info


def extract_info_from_anonymized_subgraph_path(graph_path):
    splits = graph_path.split(".txt")[0].split(os.path.sep)
    logger.debug(splits)

    info = extract_info_from_data_str(splits[2])
    info.update(extract_info_from_clusters_str(splits[-1]))

    logger.debug(info)
    # raise Exception()
    return info

def extract_info_from_clusters_path(clusters_path):
    # raise Exception(ntpath.basename(clusters_path))
    # splits = clusters_path.split(".txt")

    # raise Exception("{} - {} - {}".format(splits[0], os.path.sep, splits[0].split(os.path.sep), ))
    try:
        splits = clusters_path.split(".txt")[0].split(os.path.sep)
        logger.debug(splits)
        # raise Exception(splits)
        info = extract_info_from_data_str(splits[-4])
        info.update(extract_info_from_anony_mode_str(splits[-2]))
        info.update(extract_info_from_clusters_str(splits[-1]))
    except Exception as e:
        logger.error("path: {}".format(clusters_path))
        raise e


    return info


def get_model_name(info_loss_name, num_dimensions, args):
    """
    Get the model name.
    """
    info_loss_str = get_info_loss_full_string(info_loss_name, args)
    model_str = "{}_{}".format(info_loss_str, num_dimensions)

    return model_str


def get_model_checkpoint_path(data_name, sample, info_loss_name, num_dimensions, args):
    """
    Get model checkpoint path.
    """
    data_str = get_raw_graph_dir_str(data_name, sample)
    model_str = get_model_name(info_loss_name, num_dimensions, args)

    return os.path.join(settings.MODEL_CHECKPOINT_PATH, data_str, model_str, settings.CHECKPOINT_FILENAME)


def get_points_dir_path(data_name, sample):
    """
    Get the generated points dir path.
    """
    output_path = get_output_path(data_name, sample)
    return os.path.join(output_path, "points")

def get_points_path(data_name, sample, info_loss_name, num_dimensions, args):
    dir_path = get_points_dir_path(data_name, sample)
    model_str = get_model_name(info_loss_name, num_dimensions, args)

    return os.path.join(dir_path, model_str + ".npy")

def get_latex_table_path(exp_name, dfs, x_name, y_name, cat_name, sub_exp_name):
    setting_str = "{}-{}-{}".format(x_name, y_name, cat_name)
    data_str = "_".join([df["data"].unique()[0] for df in dfs])

    exp_str = "{}-{}".format(exp_name, sub_exp_name)

    fig_name = "{}-{}-{}".format(data_str, exp_str, setting_str).replace("#", "-")
    fig_path = putils.get_tuning_clusters_fig_dir_path(exp_name)

    path = os.path.join(fig_path, fig_name + ".tex")
    return path