from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
import glob
import argparse
import logging
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import anonygraph.utils.visualization as vutils
import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo
from anonygraph.constants import *

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


EXP_NAME = "tuning_graphs"

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    # parser.add_argument("--refresh", type=rutils.str2list(rutils.str2bool))
    parser.add_argument("--type")
    parser.add_argument("--refresh", type=rutils.str2list(rutils.str2bool))

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



def test_anonymity(df):
    invalid_clusters_df = df[df[NUM_INVALID_ANONYMITY_CLUSTERS_METRIC] > 0]

    logger.info(
        "there are {} clusters containing invalid anonymized clusters".format(
            len(invalid_clusters_df)
        )
    )

    enforcer_names = invalid_clusters_df["enforcer"].unique()
    for enforcer_name in enforcer_names:
        current_df = invalid_clusters_df[invalid_clusters_df["enforcer"] == enforcer_name]
        logger.info("enforcer: {} - {} invalid clusters".format(enforcer_name, len(current_df)))

    big_clusters_df = df[(~df["enforcer"].isin([SR_ENFORCER])) &
                         (df[NUM_BIG_CLUSTERS_METRIC] > 0)]

    logger.info(
        "there are {} clusters containing big anonymized clusters".format(
            len(big_clusters_df)
        )
    )


def get_figure_path(df, x_name, y_name, cat_name, sub_exp_name):
    data_name = df["data"].unique()[0]
    sample = df["sample"].unique()[0]

    setting_str = "{}-{}-{}".format(x_name, y_name, cat_name)
    data_str = putils.get_raw_graph_dir_str(data_name, sample)

    # sub_exp_name = "calgo"
    exp_str = "{}-{}".format(EXP_NAME, sub_exp_name)

    fig_name = "{}-{}-{}".format(data_str, exp_str, setting_str).replace("#", "-")
    fig_path = putils.get_tuning_clusters_fig_dir_path(EXP_NAME)

    path = os.path.join(fig_path, fig_name + ".pdf")
    return path

def get_latex_table_path(df, x_name, y_name, cat_name, sub_exp_name):
    data_name = df["data"].unique()[0]
    sample = df["sample"].unique()[0]

    setting_str = "{}-{}-{}".format(x_name, y_name, cat_name)
    data_str = putils.get_raw_graph_dir_str(data_name, sample)

    # sub_exp_name = "calgo"
    exp_str = "{}-{}".format(EXP_NAME, sub_exp_name)

    fig_name = "{}-{}-{}".format(data_str, exp_str, setting_str).replace("#", "-")
    fig_path = putils.get_tuning_clusters_fig_dir_path(EXP_NAME)

    path = os.path.join(fig_path, fig_name + ".tex")
    return path

def visualize_gen_settings(df, y_name):
    x_name = "gen_str"
    cat_name = "calgo_str"

    path = get_figure_path(df, x_name, y_name, cat_name, "calgo")

    df = df[df["enforcer"].isin([SR_ENFORCER])]
    df = df.sort_values(["gen_str", "calgo_str"])
    # logger.debug(df.columns)
    # logger.debug(df["calgo_str"])

    vutils.visualize_bar_chart(df, x_name, y_name, cat_name, path)


def visualize_max_dist(df, y_name, gen_str):
    cat_name = "calgo_str"
    x_name = "max_dist"

    path = get_figure_path(
        df, x_name, y_name, cat_name, "enforcer_{}".format(gen_str)
    )
    logger.debug(df.columns)

    df = df[(df["enforcer"].isin([MERGE_SPLIT_ENFORCER])) &
            (df["gen_str"].isin([gen_str]))
            # (df["calgo_str"].isin(["vac", "km#max", "hdbscan#max"]))
    ]

    df = df.sort_values([cat_name, x_name])

    vutils.visualize_bar_chart(df, x_name, y_name, cat_name, path)


def visualize_comparision(df, y_name, gen_str):
    cat_name = ALGO_KEY_COL
    x_name = "max_dist"

    path = get_figure_path(
        df, x_name, y_name, cat_name, "enforcer_{}".format(gen_str)
    )

    df = df[
        (df["enforcer"].isin([MERGE_SPLIT_ENFORCER, KMEANS_PARTITION_ENFORCER])) &
        (df["gen_str"].isin([gen_str])) &
        (df[ALGO_KEY_COL].isin(["vac-ms", "hdbscan#max-kp", "km#max-kp"]))
    ]

    df = df.sort_values([cat_name, x_name])

    clustering_keys = df[ALGO_KEY_COL].unique()
    logger.debug("algo keys: {}".format(clustering_keys))
    logger.debug(df[[y_name, x_name, cat_name, ALGO_KEY_COL, "algo_name"]])

    vutils.visualize_line_chart(df, x_name, y_name, "algo_name", path)
    # vutils.visualize_bar_chart(df, x_name, y_name, "algo_name", path)


def test_unnecessary_anony_clusters(df):
    enforcer_names = df["enforcer"].unique()

    for enforcer_name in enforcer_names:
        edf = df[df["enforcer"]==enforcer_name]
        logger.info("enforcer: {} - {} clusters".format(enforcer_name, len(edf)))
    # logger.info(len(df[df["enforcer"]=="kp"]))

    edf = df[df["enforcer"] == KMEANS_PARTITION_ENFORCER]
    calgo_names = edf["calgo"].unique()

    for calgo_name in calgo_names:
        cdf = edf[edf["calgo"] == calgo_name]
        logger.info("calgo: {} - {} clusters".format(calgo_name, len(cdf)))
    # logger.info(edf[["enforcer_str", "calgo", "calgo_args"]])

def run_visualizing_comparision(agg_df, metric_names):
    # visualize_comparision(
    #     df=agg_df, y_name=metric_names[0], gen_str="rzipf#5,50,5,2"
    # )
    for metric_name in metric_names:
        visualize_comparision(
            df=agg_df, y_name=metric_name, gen_str="zipf#5,50,5,2"
        )


def run_visualizing_max_dist(agg_df, metric_names):
    for metric_name in metric_names:
        visualize_max_dist(
            df=agg_df, y_name=metric_name, gen_str="zipf#5,50,5,2"
        )


def run_visualizing_gen_settings(agg_df, metric_names):
    for metric_name in metric_names:
        visualize_gen_settings(df=agg_df, y_name=metric_name)

def generate_max_dist_latex_table(df, y_name, gen_str):
    cat_name = "calgo_str"
    x_name = "max_dist"

    path = get_latex_table_path(
        df, x_name, y_name, cat_name, "enforcer_{}".format(gen_str)
    )
    logger.debug(df.columns)

    df = df[(df["enforcer"].isin([MERGE_SPLIT_ENFORCER])) &
            (df["gen_str"].isin([gen_str])) &
            (df["calgo_str"].isin(["hdbscan#max"]))
    ]

    df = df.sort_values([cat_name, x_name])
    logger.info(df[["max_dist", "calgo_str", "enforcer", "gen_str", y_name]])

    with open(path, 'w') as f:
        # write header
        f.write("""
        \\begin{tabularx}{\linewidth}{ c  l  y  y  y  y  y  y  y }
        & \multicolumn{1}{c}{}         & \multicolumn{3}{c}{hdbscan} & \multicolumn{3}{c}{k-medoids} & \multirow{2}{*}{VAC} \\ \cmidrule{3-5}\cmidrule{6-8}
        Data                      & \multicolumn{1}{c}{$\tau$} & min      & mean    & max     & min      & mean     & max      &                      \\\ \midrule
        """)

        # write lines
        max_dist_list = df["max_dist"].unique()
        data_name = df["data"].unique()[0]
        for idx, max_dist in enumerate(max_dist_list):
            current_df = df[df["max_dist"]==max_dist]

            line = ""

            if idx == 0:
                first_col = "\\multirow{4}{*}{" + data_name.capitalize() + "}"
            else:
                first_col = " "

            logger.debug(current_df[["max_dist", "calgo_str", y_name]])
            hmin_val = current_df[current_df["calgo_str"]=="hdbscan#min"][y_name].values[0]
            hmean_val = current_df[current_df["calgo_str"]=="hdbscan#mean"][y_name].values[0]
            hmax_val = current_df[current_df["calgo_str"]=="hdbscan#max"][y_name].values[0]
            kmin_val = current_df[current_df["calgo_str"]=="km#min"][y_name].values[0]
            kmean_val = current_df[current_df["calgo_str"]=="km#mean"][y_name].values[0]
            kmax_val = current_df[current_df["calgo_str"]=="km#max"][y_name].values[0]
            vac_val = current_df[current_df["calgo_str"]=="vac"][y_name].values[0]
            # raise Exception(hmin_val)

            line = "{first_col} & {max_dist:.2f} & {hmin:.3f} & {hmean:.3f} & {hmax:.3f} & {kmin:.3f} & {kmean:.3f} & {kmax:.3f} & {vac:.4f} \\\\ \n".format(
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

            f.write(line)
            # logger.info(max_dist)

        # write footer
        f.write("\\end{tabularx}")

    # vutils.visualize_bar_chart(df, x_name, y_name, cat_name, path)

def run_generate_max_dist_latex_table(agg_df, metric_names):
    for metric_name in metric_names:
        generate_max_dist_latex_table(agg_df, metric_name, gen_str="zipf#5,50,5,2")

def main(args):
    logger.debug(args)
    src_type = args["src_type"]
    data_names = args["data_list"]
    samples = [-1 for _ in range(len(data_names))]

    data_name = args["data"]
    sample = args["sample"]

    data_path = putils.get_tuning_exp_data_path(
        EXP_NAME, data_name, sample, args
    )

    agg_data_path = putils.get_agg_tuning_exp_data_path(
        EXP_NAME, data_name, sample, args
    )

    df = vutils.get_exp_data(
        exp_path=data_path,
        prepare_data_fn=vutils.prepare_anony_graphs_data,
        prepare_data_args={
            "data": data_name,
            "sample": sample,
        },
        workers=args["workers"],
        refresh=args["refresh"][0],
        args=args
    )

    agg_df = vutils.get_exp_data(
        exp_path=agg_data_path,
        prepare_data_fn=vutils.aggregate_clusters_data,
        prepare_data_args={
            "df": df,
            # "sample": sample,
        },
        workers=args["workers"],
        refresh=args["refresh"][1] or args["refresh"][0],
        args=args
    )

    logger.debug(agg_df)

    add_more_info(df)
    add_more_info(agg_df)

    logger.debug(df)
    logger.info("visualizing")
    logger.info("visualizing")
    metric_names = [ADM_METRIC, REMAINING_ADM_METRIC, RATIO_REMOVED_ENTITIES_COL]

    if args["type"] == "fig":
        # run_visualizing_gen_settings(agg_df, metric_names)
        # run_visualizing_max_dist(agg_df, metric_names)
        run_visualizing_comparision(agg_df, metric_names)
        # run_generate_max_dist_latex_table(agg_df, [ADM_METRIC])
    elif args["type"] == "tab":
        vutils.run_generate_gen_latex_table(agg_dfs, metric_names)



if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
