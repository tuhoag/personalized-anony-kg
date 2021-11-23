import argparse
import logging
import tensorflow as tf
import time

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.algorithms.clustering as calgo

logger = logging.getLogger(__file__)
# tf.logging.set_verbosity(tf.logging.ERROR)

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_clustering_argument(parser)
    rutils.add_k_generator_argument(parser)
    rutils.add_info_loss_argument(parser)
    rutils.add_log_argument(parser)


def main(args):
    logger.info(args)
    data_name = args["data"]
    sample = args["sample"]
    info_loss_name = args["info_loss"]
    k_generator_name = args["gen"]
    calgo_name = args["calgo"]
    outliers_handler_name = args["handler"]

    # load dist matrix
    dist_matrix = dutils.load_dist_matrix(data_name, sample, info_loss_name, args)
    logger.debug("dist matrix shape: {}".format(dist_matrix.shape))

    # load k values
    entity_id2k_dict = dutils.load_entity_id2k_dict(data_name, sample, k_generator_name, args)
    logger.debug("id2k dict: {}".format(entity_id2k_dict))

    # load relative ids
    entity_id2idx_dict = dutils.load_entity_id2idx_dict(data_name, sample)
    logger.debug("id2idx dict: {}".format(entity_id2idx_dict))

    entity_idx2id_dict = dutils.load_entity_idx2id_dict(data_name, sample)
    logger.debug("idx2id dict: {}".format(entity_idx2id_dict))

    start_time = time.time()

    algo = calgo.RawClustersGeneration(calgo_name, outliers_handler_name, args)
    clusters = algo.run(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict, args)
    logger.debug("raw clusters: {}".format(clusters))

    run_time = time.time() - start_time

    clusters_path = putils.get_raw_clusters_path(data_name, sample, k_generator_name, info_loss_name, outliers_handler_name, calgo_name, args)
    logger.info("saved raw clusters to {}".format(clusters_path))
    clusters.to_file(clusters_path)

    logger.info("finished in {}".format(run_time))

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)