import time
import logging

from anonygraph.algorithms.clustering import AnonymizedClustersGeneration
import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils

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

    # load raw clusters
    clusters = dutils.load_raw_clusters(data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, args)
    logger.debug("raw clusters: {}".format(clusters))

    start_time = time.time()
    algo_fn = AnonymizedClustersGeneration(enforcer_name, args)
    new_clusters = algo_fn.run(clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict, args)

    run_time = time.time() - start_time

    # test_minimum_clusters_size(new_clusters, entity_id2k_dict)

    anonymized_clusters_path = putils.get_anony_clusters_path(data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args)
    new_clusters.to_file(anonymized_clusters_path)
    logger.info("saved to: {}".format(anonymized_clusters_path))

    logger.info("finished in {}".format(run_time))

def test_minimum_clusters_size(clusters, entity_id2k_dict):
    invalid_clusters = []

    for cluster in clusters:
        if cluster.num_entities < cluster.get_max_k(entity_id2k_dict):
            invalid_clusters.append(cluster)


    assert len(invalid_clusters) == 0, "There are {} invalid clusters: {}".format(len(invalid_clusters), invalid_clusters)

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)