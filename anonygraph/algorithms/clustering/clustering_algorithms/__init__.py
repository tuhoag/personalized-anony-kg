from anonygraph.algorithms.clustering.clustering_algorithms.hdbscan import HDBSCANAlgorithm
import numpy as np
import logging

from .k_medoids import KMedoidsAlgorithm
from .psize import PSizeAlgorithm
from anonygraph.constants import *

logger = logging.getLogger(__name__)
def get_clustering_algo(calgo_name):
    calgo_fn_dict = {
        K_MEDOIDS_CLUSTERING_ALGORITHM: run_k_medoids_clustering,
        PSIZE_CLUSTERING_ALGORITHM: run_psize_clustering,
        HDBSCAN_CLUSTERING_ALGORITHM: run_hdbscan_clustering,
    }

    if calgo_name not in calgo_fn_dict:
        raise NotImplementedError("Unsupported {} clustering algorithm".format(calgo_name))

    return calgo_fn_dict[calgo_name]


def get_expected_k(entity_id2k_dict, args):
    # splits = args["calgo_args"].split(",")
    logger.debug(args["calgo_args"])
    mode = args["calgo_args"][0]

    k_values = list(entity_id2k_dict.values())
    if mode == "max":
        expected_k = np.max(k_values)
    elif mode == "min":
        expected_k = np.min(k_values)
    elif mode == "mean":
        expected_k = np.median(k_values)
    else:
        raise Exception("Unsupported k mean with mode: {}".format(mode))

    return expected_k

def run_k_medoids_clustering(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict, args):
    expected_k = get_expected_k(entity_id2k_dict, args)

    num_entities = dist_matrix.shape[0]
    num_clusters = int(num_entities / expected_k)

    clustering_algo = KMedoidsAlgorithm(num_clusters)
    clusters = clustering_algo.run(dist_matrix)
    return clusters


def run_hdbscan_clustering(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict, args):
    expected_k = get_expected_k(entity_id2k_dict, args)

    logger.debug("expected k: {}".format(expected_k))
    # raise Exception(expected_k)
    clustering_algo = HDBSCANAlgorithm(expected_k)
    clusters = clustering_algo.run(dist_matrix)
    return clusters


def run_psize_clustering(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict, args):
    clustering_algo = PSizeAlgorithm()
    clusters = clustering_algo.run(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict)
    return clusters