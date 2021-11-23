import glob
import itertools
import logging
import os
from time import time

from anonygraph.data import StaticGraph
import anonygraph.algorithms as algo
import anonygraph.evaluation.clusters_metrics as cmetrics
import anonygraph.evaluation.graph_metrics as gmetrics
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anonygraph.constants import *
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_exp_data(
    exp_path, prepare_data_fn, prepare_data_args, workers, refresh, args
):
    if refresh or not os.path.exists(exp_path):
        logger.info('preparing data')
        raw_data = prepare_data_fn(prepare_data_args, workers, args)

        if not os.path.exists(os.path.dirname(exp_path)):
            logger.info('creating folder: {}'.format(os.path.dirname(exp_path)))
            os.makedirs(os.path.dirname(exp_path))

        logger.debug(raw_data)

        raw_df = pd.DataFrame(raw_data)
        # df = aggregate_clusters_data(raw_df)

        logger.info('saving raw data to: {}'.format(exp_path))
        raw_df.to_csv(exp_path, index=False)

        # logger.info('saving agg data to: {}'.format(agg_exp_path))
        # df.to_csv(exp_path, index=False)
    else:
        logger.info('reading data from: {}'.format(exp_path))
        raw_df = pd.read_csv(exp_path)

        # logger.info('reading agg data from: {}'.format(agg_exp_path))
        # df = pd.read_csv(agg_exp_path)

    return raw_df

def get_cluster_quality(clusters, graph, args):
    metrics_names = cmetrics.get_all_metric_names()

    return cmetrics.calculate_quality_metrics(
        metrics_names, clusters, graph, args
    )

def get_anonymized_graph_quality(anonymized_graph, graph, args):
    metrics_names = gmetrics.get_all_metric_names()

    return gmetrics.calculate_quality_metrics(
        metrics_names, anonymized_graph, graph, args
    )


def get_graphs_quality_from_path(graph_path):
    info = putils.extract_info_from_anonymized_subgraph_path(graph_path)
    logger.debug(info)

    anonymized_subgraph = dutils.load_anonymized_graph_from_path(
        data_name=info["data"],
        sample=info["sample"],
        k_generator_name=info["gen"],
        info_loss_name=info["info_loss"],
        enforcer_name=info["enforcer"],
        calgo_name=info["calgo"],
        handler_name=info["handler_str"],
        args=info
    )

    raw_subgraph = dutils.load_raw_graph(
        info["data"], info["sample"]
    )

    quality_info = get_anonymized_graph_quality(
        anonymized_subgraph, raw_subgraph, info
    )
    info.update(quality_info)

    return info

def get_clusters_quality_from_path(clusters_path):
    clusters = algo.Clusters.from_file(clusters_path)
    info = putils.extract_info_from_clusters_path(clusters_path)


    graph = dutils.load_raw_graph(info["data"], info["sample"])
    quality_info = get_cluster_quality(clusters, graph, info)
    info.update(quality_info)

    return info

def prepare_clusters_data(data_info, num_workers, args):
    data_name, sample = data_info["data"], data_info[
        "sample"]

    clusters_dir_path = putils.get_clusters_dir_path(
        data_name, sample, "anony"
    )
    logger.debug("clusters path: {}".format(clusters_dir_path))

    clusters_paths = glob.glob(clusters_dir_path + "/*")
    logger.info("preparing data from {} clusters".format(len(clusters_paths)))
    logger.debug(clusters_paths)
    # raise Exception()
    start_time = time()
    raw_data = list(
        Parallel(n_jobs=num_workers)(
            delayed(get_clusters_quality_from_path)(path)
            for path in tqdm(clusters_paths)
        )
    )
    logger.debug(raw_data)
    logger.info(
        "finished preparing data in {} seconds".format(time() - start_time)
    )

    return raw_data

def prepare_anony_graphs_data(data_info, num_workers, args):
    data_name, sample = data_info["data"], data_info[
        "sample"]

    anony_graphs_dir_path = putils.get_anony_graphs_dir_path(
        data_name, sample
    )
    logger.debug("anony graphs path: {}".format(anony_graphs_dir_path))

    anony_graphs_paths = glob.glob(anony_graphs_dir_path + "/*")
    logger.info("preparing data from {} anony graphs".format(len(anony_graphs_paths)))
    logger.debug(anony_graphs_paths)
    # raise Exception()
    start_time = time()
    raw_data = list(
        Parallel(n_jobs=num_workers)(
            delayed(get_graphs_quality_from_path)(path)
            for path in tqdm(anony_graphs_paths)
        )
    )
    logger.debug(raw_data)
    logger.info(
        "finished preparing data in {} seconds".format(time() - start_time)
    )

    return raw_data

def remove_args_cols(key_columns):
    new_key_columns = set()
    for col_name in key_columns:
        if col_name.endswith("args"):
            continue

        new_key_columns.add(col_name)

    return new_key_columns

def aggregate_graphs_data(data_info, num_workers, args):
    df = data_info["df"]

    metrics_name = cmetrics.get_all_metric_names()
    key_columns = set(df.columns)
    key_columns.difference_update(metrics_name)
    key_columns.remove("gen_n")
    key_columns = remove_args_cols(key_columns)
    key_columns = list(key_columns)

    logger.debug("key columns: {}".format(key_columns))
    # logger.debug(df[df["calgo"] == "ps"][["calgo"]])

    # df = df.sort_values(["exp_key", "gen_n"])
    # logger.debug(df[["exp_key", "gen_n", "adm"]])
    logger.debug("nan data: {}".format(df[df.isna().any(axis=1)]))
    df.fillna("", inplace=True)

    agg_df = df.groupby(key_columns).mean()
    agg_df = agg_df.reset_index()

    logger.debug("num columns before: {} - after: {}".format(len(df.columns), len(agg_df.columns)))
    logger.debug("calgo str in df: {}".format(df["calgo_str"].unique()))
    logger.debug("calgo str in agg df: {}".format(agg_df["calgo_str"].unique()))


    # agg_df = agg_df.sort_values(["exp_key"])
    # logger.debug(agg_df[["exp_key", "gen_n", "adm", "calgo"]])
    logger.debug("len before: {} after: {}".format(len(df), len(agg_df)))
    logger.debug("calgo_str: before {} - after: {}".format(len(df["calgo_str"].unique()), len(agg_df["calgo_str"].unique())))
    # raise Exception()

    return agg_df

def aggregate_clusters_data(data_info, num_workers, args):
    df = data_info["df"]

    metrics_name = cmetrics.get_all_metric_names()
    key_columns = set(df.columns)
    key_columns.difference_update(metrics_name)
    key_columns.remove("gen_n")
    key_columns = remove_args_cols(key_columns)
    key_columns = list(key_columns)

    logger.debug("key columns: {}".format(key_columns))
    # logger.debug(df[df["calgo"] == "ps"][["calgo"]])

    # df = df.sort_values(["exp_key", "gen_n"])
    # logger.debug(df[["exp_key", "gen_n", "adm"]])
    logger.debug("nan data: {}".format(df[df.isna().any(axis=1)]))
    df.fillna("", inplace=True)

    agg_df = df.groupby(key_columns).mean()
    agg_df = agg_df.reset_index()

    logger.debug("num columns before: {} - after: {}".format(len(df.columns), len(agg_df.columns)))
    logger.debug("calgo str in df: {}".format(df["calgo_str"].unique()))
    logger.debug("calgo str in agg df: {}".format(agg_df["calgo_str"].unique()))


    # agg_df = agg_df.sort_values(["exp_key"])
    # logger.debug(agg_df[["exp_key", "gen_n", "adm", "calgo"]])
    logger.debug("len before: {} after: {}".format(len(df), len(agg_df)))
    logger.debug("calgo_str: before {} - after: {}".format(len(df["calgo_str"].unique()), len(agg_df["calgo_str"].unique())))
    # raise Exception()

    return agg_df


def save_figure(figure, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    logger.info("saving figure to: {}".format(path))
    figure.savefig(path)

def visualize_bar_chart(df, x_name, y_name, cat_name, path):
    figure = sns.barplot(data=df, y=y_name, x=x_name, hue=cat_name).get_figure()
    plt.ylabel(get_title(y_name))
    plt.xlabel(get_title(x_name))
    plt.grid(linestyle="--", axis="y", color="grey", linewidth=0.5)
    plt.legend(title=get_title(cat_name))

    if path is not None:
        save_figure(figure, path)

    plt.show()

def visualize_line_chart(df, x_name, y_name, cat_name, path):
    x_values = df[x_name].unique()
    cat_values= df[cat_name].unique()

    logger.debug("x: {} - values: {}".format(x_name, x_values))
    logger.debug("cat: {} - values: {}".format(cat_name, cat_values))

    figure = sns.lineplot(data=df, y=y_name, x=x_name, hue=cat_name, markers=True).get_figure()

    plt.ylabel(get_title(y_name))
    plt.xlabel(get_title(x_name))
    plt.grid(linestyle="--", axis="y", color="grey", linewidth=0.5)
    plt.xticks(x_values)
    plt.legend(title=get_title(cat_name))

    if path is not None:
        save_figure(figure, path)

    plt.show()

def get_title(name):
    name_dict = {
        ADM_METRIC: "Average Information Loss",
        REMAINING_ADM_METRIC: "Average Information Loss of Remaining Users",
        "ratio_removed_entities": "Ratio of Removed Users (%)",
        "gen_str": "Generation Settings",
        "new_gen_str": "Generation Settings",
        "calgo_str": "Clustering Algorithms",
        "max_dist": r"$\tau$",
        ALGO_KEY_COL: "Algorithms",
        "algo_name": "Algorithms",
    }

    return name_dict[name]


def generate_gen_settings_latex_table(dfs, y_name):
    x_name = "gen_str"
    cat_name = "calgo_str"

    path = get_latex_table_path(dfs, x_name, y_name, cat_name, "calgo")

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
                hmin_val = current_df[current_df["calgo_str"]=="hdbscan#min"][y_name].values[0]
                hmean_val = current_df[current_df["calgo_str"]=="hdbscan#mean"][y_name].values[0]
                hmax_val = current_df[current_df["calgo_str"]=="hdbscan#max"][y_name].values[0]
                kmin_val = current_df[current_df["calgo_str"]=="km#min"][y_name].values[0]
                kmean_val = current_df[current_df["calgo_str"]=="km#mean"][y_name].values[0]
                kmax_val = current_df[current_df["calgo_str"]=="km#max"][y_name].values[0]
                vac_val = current_df[current_df["calgo_str"]=="vac"][y_name].values[0]
                # raise Exception(hmin_val)

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


def run_generate_gen_latex_table(agg_dfs, metric_names):
    for metric_name in metric_names:
        generate_gen_settings_latex_table(dfs=agg_dfs, y_name=metric_name)