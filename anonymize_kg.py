import sys
import argparse
import logging

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.algorithms.generalization as gen
from anonygraph.algorithms import Clusters

logger = logging.getLogger(__file__)

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_k_generator_argument(parser)
    rutils.add_info_loss_argument(parser)
    rutils.add_clustering_argument(parser)
    rutils.add_cluster_constraint_enforcer_argument(parser)
    rutils.add_log_argument(parser)



def main(args):
    logger.info(args)
    data_name = args["data"]
    sample = args["sample"]
    info_loss_name = args["info_loss"]
    k_generator_name = args["gen"]
    calgo_name = args["calgo"]
    enforcer_name = args["enforcer"]
    # max_dist = args["max_dist"]
    handler_name = args["handler"]

    graph = dutils.load_raw_graph(data_name, sample)
    logger.debug(graph)

    clusters_path = putils.get_anony_clusters_path(data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args)
    clusters = Clusters.from_file(clusters_path)


    # load algo to generate graphs
    gen_fn = gen.GraphGeneralization()
    anony_graph = gen_fn.run(graph, clusters)
    logger.debug("final anonymized subgraph: {}".format(anony_graph))

    # save anonymized graph
    anony_graph_path = putils.get_anony_graph_path(
        data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args
    )

    anony_graph.to_edge_files(anony_graph_path)
    logger.info(
        "saved anonymized graph to {}".format(
            anony_graph_path
        )
    )

    sys.exit(0)

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)