import logging
import numpy as np
import sys

import anonygraph.utils.test as tutils
from anonygraph.algorithms import cluster, clustering
from .base_enforcer import BaseEnforcer
from anonygraph.algorithms import Cluster, Clusters, clustering as calgo
from anonygraph.constants import *
from anonygraph.algorithms.clustering.clustering_algorithms.psize import calculate_dnearest_dist_matrix, initialize_k_sequence,calculate_idx2dnearest_seq

logger = logging.getLogger(__name__)

class MergeSplitEnforcer(BaseEnforcer):
    def __init__(self, args):
        super().__init__(MERGE_SPLIT_ENFORCER, args)
        self.max_dist = float(args[0])

    def call(self, clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
        # invalid removal
        logger.info("before enforcing")
        # tutils.print_invalid_and_big_clusters(clusters, entity_id2k_dict)

        valid_clusters, removed_entity_ids = get_valid_clusters(clusters, entity_id2k_dict)
        logger.debug("valid clusters: {}".format(valid_clusters))

        # test invalid clusters
        logger.info("after removing invalid clusters")
        # tutils.print_invalid_and_big_clusters(valid_clusters, entity_id2k_dict)
        # tutils.assert_invalid_clusters(valid_clusters, entity_id2k_dict)

        k_seq = initialize_k_sequence(entity_id2idx_dict, entity_id2k_dict)
        dnearest_seq = calculate_idx2dnearest_seq(dist_matrix, k_seq)
        new_dist_matrix = calculate_dnearest_dist_matrix(dist_matrix, dnearest_seq, k_seq)
        # raise Exception()
        # merge
        merge_clusters(valid_clusters, removed_entity_ids, self.max_dist, new_dist_matrix, entity_id2idx_dict, entity_id2k_dict)

        # test invalid clusters
        logger.info("after merging users to clusters")
        tutils.print_invalid_and_big_clusters(valid_clusters, entity_id2k_dict)
        # tutils.assert_invalid_clusters(valid_clusters, entity_id2k_dict)

        # # split clusters
        split_big_clusters(valid_clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict)

        # test invalid clusters
        logger.info("after spliting big clusters")
        # tutils.print_invalid_and_big_clusters(valid_clusters, entity_id2k_dict)
        # tutils.assert_invalid_and_big_clusters(valid_clusters, entity_id2k_dict)

        return valid_clusters

def get_valid_clusters(clusters, entity_id2k_dict):
    valid_clusters = Clusters()
    entity_ids = set()

    # logger.debug("initial valid clusters: {}".format(valid_clusters))
    for cluster in clusters:
        max_k = 0

        for entity_id in cluster:
            entity_k = entity_id2k_dict[entity_id]
            max_k = max(max_k, entity_k)

        if cluster.num_entities >= max_k:
            new_cluster = cluster.copy()
            valid_clusters.add_cluster(new_cluster)
        else:
            entity_ids.update(cluster.entity_ids)

        # logger.debug("valid clusters: {}".format(valid_clusters))
        # raise Exception()

    # logger.debug("entity ids: {}".format(entity_ids))
    # logger.debug("clusters: {}".format(valid_clusters))
    return valid_clusters, entity_ids


def merge_clusters(clusters, entity_ids, max_dist, dist_matrix, entity_id2idx_dict, entity_id2k_dict):
    real_max_dist = calculate_real_max_dist(max_dist, dist_matrix)

    # for entity_id in entity_ids:
    for entity_id in entity_ids:
        closest_cluster = find_closest_cluster(entity_id, clusters, real_max_dist, dist_matrix,  entity_id2idx_dict, entity_id2k_dict)

        if closest_cluster is None:
            # create new cluster
            closest_cluster = Cluster()

        # add to cluster
        closest_cluster.add_entity(entity_id)

def calculate_real_max_dist(max_dist, dist_matrix):
    # logger.debug(dist_matrix)
    max_pair_dist = np.max(dist_matrix)
    np.fill_diagonal(dist_matrix, max_pair_dist)
    min_pair_dist = np.min(dist_matrix)

    real_max_dist = (max_pair_dist - min_pair_dist) * max_dist + min_pair_dist

    logger.debug("max dist: {} -> real max dist: {} (max:{} - min:{})".format(max_dist, real_max_dist, max_pair_dist, min_pair_dist))
    return real_max_dist

def find_closest_cluster(entity_id, clusters, max_dist, dist_matrix, id2idx_dict, id2k_dict):
    entity_k = id2k_dict[entity_id]

    closest_cluster = None
    smallest_dist = sys.maxsize

    # logger.debug(clusters)
    # raise Exception()
    for cluster in clusters:
        distance = calculate_distance_from_entity_to_cluster(entity_id, cluster, dist_matrix, id2idx_dict)
        k_values = [id2k_dict[entity_id] for entity_id in cluster]
        max_k = max(max(k_values), entity_k)

        # logger.debug("k values: {} - entity_k: {} - max: {}  - cluster size: {} - cluster: {} ".format(k_values, entity_k, max_k, cluster.num_entities, cluster))

        if distance > max_dist or cluster.num_entities + 1 < max_k:
            continue

        if distance < smallest_dist:
            smallest_dist = distance
            closest_cluster = cluster

        # tutils.assert_invalid_clusters(clusters, id2k_dict)

    return closest_cluster

def calculate_distance_from_entity_to_cluster(entity_id, cluster, dist_matrix, entity_id2idx_dict):
    entity_idx = entity_id2idx_dict[entity_id]
    logger.debug(cluster)
    cluster_entity_idxes = [entity_id2idx_dict[cluster_entity_id] for cluster_entity_id in cluster]

    distances = dist_matrix[entity_idx, cluster_entity_idxes]
    distance = max(distances)
    return distance


def split_big_clusters(clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
    index = 0

    count = 0
    while(index < len(clusters)):
        # logger.info("{}/{} clusters".format(index + 1, len(clusters)))
        current_cluster = clusters.pop(index)

        max_k = max([entity_id2k_dict[entity_id] for entity_id in current_cluster])

        if len(current_cluster) >= max_k * 2:
            new_clusters_list = split_big_cluster(current_cluster, max_k, dist_matrix, entity_id2idx_dict, entity_idx2id_dict)

            for new_cluster in new_clusters_list:
                logger.debug(new_cluster)

                # raise Exception()
                clusters.insert(index, new_cluster)

            logger.info("[{}/{}] break cluster of size {} into {}".format(index + 1, len(clusters), current_cluster, len(new_clusters_list)))

            # if count == 1:
            #     raise Exception()
            # else:
            #     count += 1
        else:
            clusters.insert(index, current_cluster)
            index += 1
            logger.info("[{}/{}] add cluster: {}".format(index + 1, len(clusters), current_cluster))

        # logger.info("{}/{} clusters".format(index + 1, len(clusters)))

def split_big_cluster(big_cluster, min_size, dist_matrix, entity_id2idx_dict, entity_idx2id_dict):
    num_clusters = int(len(big_cluster) / min_size)
    logger.debug("split {} to {} clusters".format(big_cluster, num_clusters))

    algo_fn = calgo.SameSizeKMedoidsClustering(num_clusters)
    cluster_entity_ids = big_cluster.to_list()
    cluster_entity_idxes = [entity_id2idx_dict[entity_id] for entity_id in big_cluster]
    cluster_dist_matrix = generate_distance_matrix(cluster_entity_idxes, dist_matrix)

    logger.debug("entity ids: {} - idxes: {}".format(cluster_entity_ids, cluster_entity_idxes))
    entity_idxes_list = algo_fn.run(cluster_dist_matrix)

    new_clusters_list = []
    for entity_idxes in entity_idxes_list:
        raw_entity_idxes = [cluster_entity_idxes[entity_idx] for entity_idx in entity_idxes]
        new_cluster = Cluster.from_iter(map(lambda entity_idx: entity_idx2id_dict[entity_idx], raw_entity_idxes))
        new_clusters_list.append(new_cluster)

    logger.debug("big cluster: {} - new clusters list: {}".format(big_cluster, new_clusters_list))
    # raise Exception()
    return new_clusters_list

def generate_distance_matrix(entity_idxes, dist_matrix):
    result = np.zeros((len(entity_idxes), len(entity_idxes)))

    for entity1_idx, raw_entity1_idx in enumerate(entity_idxes):
        for entity2_idx, raw_entity2_idx in enumerate(entity_idxes):
            result[entity1_idx, entity2_idx] = dist_matrix[raw_entity1_idx, raw_entity2_idx]

    return result