import argparse
import glob
import itertools
import logging
import os
from time import time

import matplotlib.pyplot as plt
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

EXP_NAME = "tunning"
TUNNING_GRAPH_EXP_NAME = "tuning_graphs"
TUNNING_CLUSTERS_EXP_NAME = "tuning_clusters"


def add_arguments(parser):
    # rutils.add_data_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--data_list", type=rutils.str2list(str))
    parser.add_argument("--refresh", type=rutils.str2list(rutils.str2bool))
    parser.add_argument("--src_type")
    parser.add_argument("--exp_names", type=rutils.str2list(str))



RATIO_REMOVED_ENTITIES_COL = "ratio_removed_entities"


def add_more_info(df):
    df[RATIO_REMOVED_ENTITIES_COL] = df[REMOVED_ENTITIES_METRIC] / df[RAW_ENTITIES_METRIC]
    df["calgo_title"] = df["calgo_str"]

    logger.debug(df["calgo_str"].unique())
    df[ALGO_KEY_COL] = df["calgo_str"] + "-" + df["enforcer"]
    logger.debug(df[ALGO_KEY_COL].unique())
    df["algo_name"] = ""
    df.loc[df[ALGO_KEY_COL]=="vac-ms", "algo_name"] = "PCKGA"
    df.loc[df[ALGO_KEY_COL]=="hdbscan#max-kp", "algo_name"] = "CKGA(hdbscan)"
    df.loc[df[ALGO_KEY_COL]=="km#max-kp", "algo_name"] = "CKGA(km)"

    logger.debug(df["algo_name"].unique())
    # change rzipf to zipf
    # change rtd to te
    # df["new_gen_str"]




def get_latex_table_path(dfs, x_name, y_name, cat_name, exp_name, sub_exp_name):
    setting_str = "{}-{}-{}".format(x_name, y_name, cat_name)
    data_str = "_".join([df["data"].unique()[0] for df in dfs])

    exp_str = "{}-{}".format(exp_name, sub_exp_name)

    fig_name = "{}-{}-{}".format(data_str, exp_str, setting_str).replace("#", "-")
    fig_path = putils.get_tuning_clusters_fig_dir_path(exp_name)

    path = os.path.join(fig_path, fig_name + ".tex")
    return path



def main(args):
    logger.debug(args)
    src_type = args["src_type"]
    data_names = args["data_list"]
    exp_types = args["exp_names"]
    samples = [-1 for _ in range(len(data_names))]

    logger.debug("data_names: {}".format(data_names))
    logger.debug("samples: {}".format(samples))

    # global EXP_NAME
    if src_type in ["c", "clusters"]:
        exp_name = TUNNING_CLUSTERS_EXP_NAME
        prepare_data_fn = vutils.prepare_clusters_data
        prepare_agg_data_fn = vutils.aggregate_clusters_data

    elif src_type in ["g", "graphs"]:
        exp_name = TUNNING_GRAPH_EXP_NAME
        prepare_data_fn = vutils.prepare_anony_graphs_data
        prepare_agg_data_fn = vutils.aggregate_clusters_data
    else:
        raise Exception()

    logger.debug("exp_name: {}".format(exp_name))

    data_paths = [
        putils.get_tuning_exp_data_path(
            exp_name, data_name, sample, args
        ) for data_name, sample in zip(data_names, samples)
    ]

    agg_data_paths = [
        putils.get_agg_tuning_exp_data_path(
            exp_name, data_name, sample, args
        ) for data_name, sample in zip(data_names, samples)
    ]

    logger.debug("data_paths: {}".format(data_paths))
    logger.debug("agg_data_paths: {}".format(agg_data_paths))
    logger.debug("prepare_data_fn: {}".format(prepare_data_fn))
    logger.debug("prepare_agg_data_fn: {}".format(prepare_agg_data_fn))

    dfs = [ vutils.get_exp_data(
            exp_path=data_path,
            prepare_data_fn=prepare_data_fn,
            prepare_data_args={
                "data": data_name,
                "sample": sample,
            },
            workers=args["workers"],
            refresh=args["refresh"][0],
            args=args
        ) for data_name, sample, data_path in zip(data_names, samples, data_paths)
    ]

    agg_dfs = [ vutils.get_exp_data(
            exp_path=agg_data_path,
            prepare_data_fn=prepare_agg_data_fn,
            prepare_data_args={
                "df": df,
                # "sample": sample,
            },
            workers=args["workers"],
            refresh=args["refresh"][1] or args["refresh"][0],
            args=args
        ) for data_name, sample, agg_data_path, df in zip(data_names, samples, agg_data_paths, dfs)
    ]


    for agg_df, df in zip(agg_dfs, dfs):
        add_more_info(agg_df)
        add_more_info(df)

    # logger.debug(df)
    logger.info("visualizing")
    metric_names = [ADM_METRIC, RATIO_REMOVED_ENTITIES_COL]

    for exp_type in exp_types:
        if exp_type == "vac":
            run_generate_gen_latex_table(agg_dfs, metric_names)
        elif exp_type == "ms":
            run_generate_max_dist_latex_table(agg_dfs, metric_names)
        elif exp_type == "compare":
            run_generate_comparision(agg_dfs, metric_names)
        else:
            raise Exception("Unsupported exp type: {}".format(exp_types))


def run_generate_comparision(agg_dfs, metric_names):
    for metric_name in metric_names:
        generate_comparision(
            dfs=agg_dfs, y_name=metric_name, gen_str="zipf#5,50,5,2"
        )

def generate_comparision(dfs, y_name, gen_str):
    cat_name = ALGO_KEY_COL
    x_name = "max_dist"

    path = get_latex_table_path(
        dfs, x_name, y_name, cat_name, "compare", "enforcer_{}".format(gen_str)
    )

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    new_dfs = []
    for df in dfs:
        new_df = df[
            (df["enforcer"].isin([MERGE_SPLIT_ENFORCER, KMEANS_PARTITION_ENFORCER])) &
            (df["gen_str"].isin([gen_str])) &
            (df[ALGO_KEY_COL].isin(["vac-ms", "hdbscan#max-kp", "km#max-kp"]))
        ]

        new_df = new_df.sort_values([cat_name, x_name])

        new_dfs.append(new_df)

    clustering_keys = new_dfs[0][ALGO_KEY_COL].unique()
    logger.debug("algo keys: {}".format(clustering_keys))
    logger.debug(df[[y_name, x_name, cat_name, ALGO_KEY_COL, "algo_name"]])

    with open(path, 'w') as f:
        # write header
        f.writelines("""
        \\begin{table}[]
        \\caption{}\\label{}
        \\footnotesize
        \\begin{tabularx}{\linewidth}{clyyyyy}\n""")

        max_dist_list = []
        for new_df in new_dfs:
            max_dist_list.extend(new_df["max_dist"].unique())
        max_dist_list = sorted(set(max_dist_list))
        max_dist_list_str = "&".join([" {:.2f} ".format(max_dist) for max_dist in max_dist_list])
        # raise Exception(max_dist_list_str)
        f.writelines("Data & Algorithm & {} \\\\ \midrule\n".format(max_dist_list_str))

        # write lines
        for idf, new_df in enumerate(new_dfs):
            algo_names = new_df["algo_name"].unique()
            data_name = new_df["data"].unique()[0]
            for idx, algo_name in enumerate(algo_names):
                current_df = new_df[new_df["algo_name"]==algo_name]

                line = ""

                if idx == 0:
                    first_col = "\\multirow{3}{*}{" + data_name.capitalize() + "}"
                else:
                    first_col = " "

                logger.debug(current_df[["gen_str", "calgo_str", y_name]])

                new_algo_str = algo_name
                v0 = current_df[current_df["max_dist"]==0][y_name].values[0]
                v025 = current_df[current_df["max_dist"]==0.25][y_name].values[0]
                v05 = current_df[current_df["max_dist"]==0.5][y_name].values[0]
                v075 = current_df[current_df["max_dist"]==0.75][y_name].values[0]
                v1 = current_df[current_df["max_dist"]==1][y_name].values[0]

                line = "{first_col}  & {algo_str} & {v0:.3f} & {v025:.3f} & {v05:.3f} & {v075:.3f} & {v1:.3f} \\\\".format(
                    first_col=first_col,
                    algo_str=new_algo_str,
                    v0=v0,
                    v025=v025,
                    v05=v05,
                    v075=v075,
                    v1=v1,
                )

                if idx == len(algo_names) - 1:
                    line += "\\midrule \n"
                else:
                    line += "\n"
                f.write(line)

        # write footer
        f.write("""\\end{tabularx}
        \end{table}""")

def get_value(df, col_name, value, y_name):
    logger.debug("{} - values: {}".format(value, df[df[col_name]==value][y_name].values))
    if len(df[df[col_name]==value][y_name].values) > 0:
        value = df[df[col_name]==value][y_name].values[0]
    else:
        value = -1

    return value

def generate_gen_settings_latex_table(dfs, y_name):
    x_name = "gen_str"
    cat_name = "calgo_str"

    path = get_latex_table_path(dfs, x_name, y_name, cat_name, "vac", "calgo")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    new_dfs = []
    for df in dfs:
        new_df = df[df["enforcer"].isin([SR_ENFORCER])
        ]
            # (df["calgo_str"].isin(["hdbscan#max"]))
        new_df = new_df.sort_values(["gen_str", "calgo_str"])

        new_dfs.append(new_df)

    logger.info(df[["max_dist", "calgo_str", "enforcer", "gen_str", y_name]])

    with open(path, 'w') as f:
        # write header
        f.write("""
        \\begin{table}[]
    \\caption{}\\label{}
    \\footnotesize
        \\begin{tabularx}{\linewidth}{ c  l  y  y  y  y  y  y  y }
        & \multicolumn{1}{c}{}         & \multicolumn{3}{c}{hdbscan} & \multicolumn{3}{c}{k-medoids} & \multirow{2}{*}{VAC} \\\\ \cmidrule{3-5}\cmidrule{6-8}
        Data                      & \multicolumn{1}{c}{Settings} & min      & mean    & max     & min      & mean     & max      &                      \\\\ \midrule
        """)

        # write lines
        for idf, new_df in enumerate(new_dfs):
            gen_list = new_df["gen_str"].unique()
            data_name = new_df["data"].unique()[0]
            for idx, gen_str in enumerate(gen_list):
                current_df = new_df[new_df["gen_str"]==gen_str]

                line = ""

                if idx == 0:
                    first_col = "\\multirow{4}{*}{" + data_name.capitalize() + "}"
                else:
                    first_col = " "

                logger.debug(current_df[["gen_str", "calgo_str", y_name]])

                new_gen_str = "${0}$".format(gen_str.replace("#", "\#"))
                hmin_val = get_value(current_df, "calgo_str", "hdbscan#min", y_name)
                hmean_val = get_value(current_df, "calgo_str", "hdbscan#mean", y_name)
                hmax_val = get_value(current_df, "calgo_str", "hdbscan#max", y_name)
                kmin_val = get_value(current_df, "calgo_str", "km#min", y_name)
                kmean_val = get_value(current_df, "calgo_str", "km#mean", y_name)
                kmax_val = get_value(current_df, "calgo_str", "km#max", y_name)
                vac_val = get_value(current_df, "calgo_str", "vac", y_name)
                # raise Exception(hmin_val)
                logger.debug(vac_val)

                line = "{first_col} & {gen_str} & {hmin:.3f} & {hmean:.3f} & {hmax:.3f} & {kmin:.3f} & {kmean:.3f} & {kmax:.3f} & {vac:.4f} \\\\".format(
                    first_col=first_col,
                    gen_str=new_gen_str,
                    hmin=hmin_val,
                    hmean=hmean_val,
                    hmax=hmax_val,
                    kmin=kmin_val,
                    kmean=kmean_val,
                    kmax=kmax_val,
                    vac=vac_val
                )

                if idx == len(gen_list) - 1:
                    line += "\\midrule \n"
                else:
                    line += "\n"
                f.write(line)
                # logger.info(max_dist)

        # write footer
        f.write("""\\end{tabularx}
        \end{table}""")

def generate_max_dist_latex_table(dfs, y_name, gen_str):
    cat_name = "calgo_str"
    x_name = "max_dist"

    path = get_latex_table_path(
        dfs, x_name, y_name, cat_name, "ms", "enforcer_{}".format(gen_str)
    )

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    new_dfs = []
    for df in dfs:
        new_df = df[(df["enforcer"].isin([MERGE_SPLIT_ENFORCER])) &
            (df["gen_str"].isin([gen_str]))
            # (df["calgo_str"].isin(["hdbscan#max"]))
        ]
        new_df = new_df.sort_values([cat_name, x_name])

        new_dfs.append(new_df)

    logger.info(df[["max_dist", "calgo_str", "enforcer", "gen_str", y_name]])

    with open(path, 'w') as f:
        # write header
        f.write("""
        \\begin{table}[]
        \\caption{}\\label{}
        \\footnotesize
        \\begin{tabularx}{\linewidth}{ c  l  y  y  y  y  y  y  y }
        & \multicolumn{1}{c}{}         & \multicolumn{3}{c}{hdbscan} & \multicolumn{3}{c}{k-medoids} & \multirow{2}{*}{VAC} \\\\ \cmidrule{3-5}\cmidrule{6-8}
        Data                      & \multicolumn{1}{c}{$\\tau$} & min      & mean    & max     & min      & mean     & max      &                      \\\\ \midrule
        """)

        # write lines
        for idf, new_df in enumerate(new_dfs):
            max_dist_list = new_df["max_dist"].unique()
            data_name = new_df["data"].unique()[0]
            for idx, max_dist in enumerate(max_dist_list):
                current_df = new_df[new_df["max_dist"]==max_dist]

                line = ""

                if idx == 0:
                    first_col = "\\multirow{5}{*}{" + data_name.capitalize() + "}"
                else:
                    first_col = " "

                logger.debug(current_df[["max_dist", "calgo_str", y_name]])
                hmin_val = get_value(current_df, "calgo_str", "hdbscan#min", y_name)
                hmean_val = get_value(current_df, "calgo_str", "hdbscan#mean", y_name)
                hmax_val = get_value(current_df, "calgo_str", "hdbscan#max", y_name)
                kmin_val = get_value(current_df, "calgo_str", "km#min", y_name)
                kmean_val = get_value(current_df, "calgo_str", "km#mean", y_name)
                kmax_val = get_value(current_df, "calgo_str", "km#max", y_name)
                vac_val = get_value(current_df, "calgo_str", "vac", y_name)

                # hmin_val = current_df[current_df["calgo_str"]=="hdbscan#min"][y_name].values[0]
                # hmean_val = current_df[current_df["calgo_str"]=="hdbscan#mean"][y_name].values[0]
                # hmax_val = current_df[current_df["calgo_str"]=="hdbscan#max"][y_name].values[0]
                # kmin_val = current_df[current_df["calgo_str"]=="km#min"][y_name].values[0]
                # kmean_val = current_df[current_df["calgo_str"]=="km#mean"][y_name].values[0]
                # kmax_val = current_df[current_df["calgo_str"]=="km#max"][y_name].values[0]
                # vac_val = current_df[current_df["calgo_str"]=="vac"][y_name].values[0]
                # raise Exception(hmin_val)

                line = "{first_col} & {max_dist:.2f} & {hmin:.3f} & {hmean:.3f} & {hmax:.3f} & {kmin:.3f} & {kmean:.3f} & {kmax:.3f} & {vac:.4f} \\\\".format(
                    first_col=first_col,
                    max_dist=max_dist,
                    hmin=hmin_val,
                    hmean=hmean_val,
                    hmax=hmax_val,
                    kmin=kmin_val,
                    kmean=kmean_val,
                    kmax=kmax_val,
                    vac=vac_val
                )

                if idx == len(max_dist_list) - 1:
                    line += "\\midrule \n"
                else:
                    line += "\n"
                f.write(line)
                # logger.info(max_dist)

        # write footer
        f.write("""\\end{tabularx}
        \end{table}""")


def run_generate_max_dist_latex_table(agg_dfs, metric_names):
    for metric_name in metric_names:
        generate_max_dist_latex_table(agg_dfs, metric_name, gen_str="zipf#5,50,5,2")


def run_generate_gen_latex_table(agg_dfs, metric_names):
    for metric_name in metric_names:
        generate_gen_settings_latex_table(dfs=agg_dfs, y_name=metric_name)

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
