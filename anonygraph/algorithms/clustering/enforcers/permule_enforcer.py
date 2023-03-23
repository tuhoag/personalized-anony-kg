import itertools
import logging
import numpy as np
import sys
from sortedcontainers import SortedList

import anonygraph.utils.test as tutils
from anonygraph.algorithms import cluster, clustering
from .base_enforcer import BaseEnforcer
from anonygraph.algorithms import Cluster, Clusters, clustering as calgo
from anonygraph.constants import *
from anonygraph.algorithms.clustering.clustering_algorithms.psize import calculate_dnearest_dist_matrix, initialize_k_sequence,calculate_idx2dnearest_seq

logger = logging.getLogger(__name__)

class PermuleEnforcer(BaseEnforcer):
    def __init__(self, args):
        super().__init__(PERMULE_ENFORCER, args)

    def call(self, clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
        # invalid removal
        logger.info("before enforcing")
        # tutils.print_invalid_and_big_clusters(clusters, entity_id2k_dict)

        logger.info("finding valid and invalid clusters")
        valid_clusters_ids, invalid_clusters_ids = get_valid_and_invalid_clusters_ids(clusters, entity_id2k_dict)
        logger.debug("clusters: {}".format(clusters))
        logger.debug("valid clusters: {}".format(valid_clusters_ids))
        logger.debug("invalid clusters: {}".format(invalid_clusters_ids))
        # for each invalid cluster, find its nearest one to merge


        # tutils.print_invalid_and_big_clusters(valid_clusters, entity_id2k_dict)
        # tutils.assert_invalid_clusters(valid_clusters, entity_id2k_dict)

        k_seq = initialize_k_sequence(entity_id2idx_dict, entity_id2k_dict)
        dnearest_seq = calculate_idx2dnearest_seq(dist_matrix, k_seq)
        new_dist_matrix = calculate_dnearest_dist_matrix(dist_matrix, dnearest_seq, k_seq)
        # raise Exception()
        # merge

        logger.info("merging invalid clusters")
        new_clusters = merge_invalid_clusters(clusters, invalid_clusters_ids, new_dist_matrix, entity_id2idx_dict, entity_id2k_dict)

        # test invalid clusters
        logger.info("after merging users to clusters")
        tutils.print_invalid_and_big_clusters(valid_clusters_ids, entity_id2k_dict)
        # tutils.assert_invalid_clusters(valid_clusters, entity_id2k_dict)

        return new_clusters

def get_valid_and_invalid_clusters_ids(clusters, entity_id2k_dict):
    max_k = max(entity_id2k_dict.values())
    valid_clusters_ids = []
    invalid_clusters_ids = []

    # logger.debug("initial valid clusters: {}".format(valid_clusters))
    for cid, cluster in enumerate(clusters):
        new_cluster = cluster.copy()
        if cluster.num_entities >= max_k:
            valid_clusters_ids.append(cid)
        else:
            invalid_clusters_ids.append(cid)

        # logger.debug("valid clusters: {}".format(valid_clusters))
        # raise Exception()

    # logger.debug("entity ids: {}".format(entity_ids))
    # logger.debug("clusters: {}".format(valid_clusters))
    return valid_clusters_ids, invalid_clusters_ids


def merge_invalid_clusters(clusters, invalid_clusters_ids, dist_matrix, entity_id2idx_dict, entity_id2k_dict):
    new_clusters = [cluster.copy() for cluster in clusters]

    new_valid_clusters_ids = []
    sorted_clusters_dist = SortedList()
    num_invalid_clusters = len(invalid_clusters_ids)
    max_k = max(entity_id2k_dict.values())

    logger.info("finding nearest clusters for {}/{} invalid ones".format(len(invalid_clusters_ids), len(new_clusters)))

    for idx, cid1 in enumerate(invalid_clusters_ids):
        logger.info("found nearest cluster for {}/{}".format(idx, len(invalid_clusters_ids)))
        cluster1 = new_clusters[cid1]
        min_dist = sys.maxsize
        min_cluster_id = None

        for cid2, cluster2 in enumerate(new_clusters):
            # cluster2 = clusters[cidx2]
            if cid2 <= cid1:
                continue

            dist = calculate_distance_between_clusters(cluster1, cluster2, dist_matrix, entity_id2idx_dict)

            if dist < min_dist:
                min_dist = dist
                min_cluster_id = cid2


        sorted_clusters_dist.add((dist, cid1, min_cluster_id))

    new_invalid_clusters_ids = set(invalid_clusters_ids)
    # new_invalid_clusters_ids.extend(invalid_clusters_ids)

    logger.debug("sorted clusters dist: {}".format(sorted_clusters_dist))
    availability = [True for _ in new_clusters]

    logger.info("merging invalid clusters for max_k={}".format(max_k))
    while len(new_invalid_clusters_ids) > 0:
        logger.info("remaining invalid clusters: {} - {}".format(len(new_invalid_clusters_ids), len(sorted_clusters_dist)))

        logger.debug("new_clusters: {}".format(new_clusters))
        logger.debug("new_valid_clusters_ids: {}".format(new_valid_clusters_ids))
        logger.debug("new_invalid_clusters_ids: {}".format(new_invalid_clusters_ids))
        # logger.debug(sorted_clusters_dist)
        # logger.debug("availability: {}".format(availability))
        logger.debug("num_invalid_clusters: {}".format(num_invalid_clusters))

        smallest_dist, cid1, cid2 = sorted_clusters_dist.pop(0)

        # logger.debug("smallest_dist: {} - cidx1: {} - cidx2: {}".format(smallest_dist, cidx1, cidx2))

        if availability[cid1] and availability[cid2]:
            c1 = new_clusters[cid1]
            c2 = new_clusters[cid2]

            cluster = Cluster.from_iter(itertools.chain(c1, c2))
            logger.debug("c1({}): {} - c2({}): {} - c: {}".format(cid1, c1, cid2, c2, cluster))

            availability[cid1] = False
            availability[cid2] = False


            if cid1 in new_invalid_clusters_ids:
                new_invalid_clusters_ids.remove(cid1)

            if cid2 in new_invalid_clusters_ids:
                new_invalid_clusters_ids.remove(cid2)


            new_clusters.append(cluster)
            availability.append(True)
            new_cluster_idx = len(new_clusters) - 1

            num_invalid_clusters -= 1

            if len(c2) < max_k:
                num_invalid_clusters -= 1
                # new_invalid_clusters_ids.remove(cid2)

            if len(cluster) < max_k:
                # new_valid_clusters.append(cluster)
                num_invalid_clusters += 1
                new_invalid_clusters_ids.add(new_cluster_idx)
            # new_clusters.append(cluster)
            # new_invalid_clusters_ids.append(cluster)
            # availability.append(True)

            logger.debug("removing invalid dists")
            idx2 = 0
            selected_cluster_ids = [cid1, cid2]

            while idx2 < len(sorted_clusters_dist):
                _, cid1_new, cid2_new = sorted_clusters_dist[idx2]
                if cid1_new in selected_cluster_ids or cid2_new in selected_cluster_ids:
                    sorted_clusters_dist.pop(idx2)

                idx2 += 1

            logger.debug("updating distances")
            for idx2 in range(len(new_clusters) - 1):
                if availability[idx2]:
                    logger.debug("added distance: {}".format((dist, idx2, new_cluster_idx)))
                    dist = calculate_distance_between_clusters(new_clusters[new_cluster_idx], new_clusters[idx2], dist_matrix, entity_id2idx_dict)

                    sorted_clusters_dist.add((dist, new_cluster_idx, idx2))

    new_available_clusters = Clusters()
    for cid, cluster in enumerate(new_clusters):
        if availability[cid]:
            new_available_clusters.add_cluster(cluster)

    logger.debug(new_available_clusters)
    # raise Exception()
    return new_available_clusters

def calculate_distance_between_clusters(invalid_cluster1, invalid_cluster2, dist_matrix, entity_id2idx_dict):
    dist = 0
    for entity1_id in invalid_cluster1:
        # for entity2_id in invalid_cluster2:
        entity_dist = calculate_distance_from_entity_to_cluster(entity1_id, invalid_cluster2, dist_matrix, entity_id2idx_dict)
        dist = max(entity_dist, dist)

    return dist


def calculate_distance_from_entity_to_cluster(entity_id, cluster, dist_matrix, entity_id2idx_dict):
    entity_idx = entity_id2idx_dict[entity_id]
    # logger.debug(cluster)
    cluster_entity_idxes = [entity_id2idx_dict[cluster_entity_id] for cluster_entity_id in cluster]

    distances = dist_matrix[entity_idx, cluster_entity_idxes]
    distance = max(distances)
    return distance

