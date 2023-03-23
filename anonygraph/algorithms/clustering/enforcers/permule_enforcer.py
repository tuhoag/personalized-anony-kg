import itertools
import logging
import numpy as np
import sys
from sortedcontainers import SortedList
from tqdm import tqdm

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
        # valid_clusters_ids, invalid_clusters_ids = get_valid_and_invalid_clusters_ids(clusters, entity_id2k_dict)
        # logger.debug("clusters: {}".format(clusters))
        # logger.debug("valid clusters: {}".format(valid_clusters_ids))
        # logger.debug("invalid clusters: {}".format(invalid_clusters_ids))

        # tutils.print_invalid_and_big_clusters(valid_clusters, entity_id2k_dict)
        # tutils.assert_invalid_clusters(valid_clusters, entity_id2k_dict)

        k_seq = initialize_k_sequence(entity_id2idx_dict, entity_id2k_dict)
        dnearest_seq = calculate_idx2dnearest_seq(dist_matrix, k_seq)
        new_dist_matrix = calculate_dnearest_dist_matrix(dist_matrix, dnearest_seq, k_seq)
        # raise Exception()
        # merge

        logger.info("merging invalid clusters")
        new_clusters = merge_invalid_clusters(clusters, new_dist_matrix, entity_id2idx_dict, entity_id2k_dict)

        # test invalid clusters
        logger.info("after merging users to clusters")
        # tutils.print_invalid_and_big_clusters(valid_clusters_ids, entity_id2k_dict)
        # tutils.assert_invalid_clusters(valid_clusters, entity_id2k_dict)

        return new_clusters

def get_valid_and_invalid_clusters_ids(clusters, entity_id2k_dict):
    max_k = max(entity_id2k_dict.values())
    valid_clusters_ids = []
    invalid_clusters_ids = set()

    # logger.debug("initial valid clusters: {}".format(valid_clusters))
    for cid, cluster in enumerate(clusters):
        new_cluster = cluster.copy()
        if cluster.num_entities >= max_k:
            valid_clusters_ids.append(cid)
        else:
            invalid_clusters_ids.add(cid)

        # logger.debug("valid clusters: {}".format(valid_clusters))
        # raise Exception()

    # logger.debug("entity ids: {}".format(entity_ids))
    # logger.debug("clusters: {}".format(valid_clusters))
    return max_k, valid_clusters_ids, invalid_clusters_ids

def merge_invalid_clusters(clusters, dist_matrix, entity_id2idx_dict, entity_id2k_dict):
    max_k, valid_cluster_ids, invalid_cluster_ids = get_valid_and_invalid_clusters_ids(clusters, entity_id2k_dict)

    all_clusters = [cluster.copy() for cluster in clusters]
    available_cluster_ids = set(range(len(clusters)))

    logger.info("finding nearest clusters for {}/{} invalid ones".format(len(invalid_cluster_ids), len(all_clusters)))

    cache_dist = {}
    while(len(invalid_cluster_ids) > 0):
        logger.info("num invalid clusters: {}".format(len(invalid_cluster_ids)))
        nearest_cluster_ids = find_
        min_dist = sys.maxsize
        nearest_cluster_ids = None

        for c_in_id in invalid_cluster_ids:
            for c_id in available_cluster_ids:
                if c_id == c_in_id:
                    continue

                if c_in_id < c_id:
                    key = (c_in_id, c_id)
                else:
                    key = (c_id, c_in_id)
                dist = cache_dist.get(key)
                if dist is None:
                    dist = calculate_distance_between_clusters(all_clusters[c_in_id], all_clusters[c_id], dist_matrix, entity_id2idx_dict)
                    cache_dist[key] = dist

                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster_ids = (c_in_id, c_id)

        new_cluster = Cluster.from_iter(itertools.chain(all_clusters[nearest_cluster_ids[0]], all_clusters[nearest_cluster_ids[1]]))
        new_cluster_id = len(all_clusters)

        all_clusters.append(new_cluster)
        available_cluster_ids.add(new_cluster_id)

        available_cluster_ids.remove(nearest_cluster_ids[0])
        available_cluster_ids.remove(nearest_cluster_ids[1])

        if len(all_clusters[nearest_cluster_ids[0]]) < max_k:
            invalid_cluster_ids.remove(nearest_cluster_ids[0])

        if len(all_clusters[nearest_cluster_ids[1]]) < max_k:
            invalid_cluster_ids.remove(nearest_cluster_ids[1])

        if len(new_cluster) < max_k:
            invalid_cluster_ids.add(new_cluster_id)

    final_clusters = Clusters()
    for c_id in available_cluster_ids:
        final_clusters.add_cluster(all_clusters[c_id])

    logger.debug(final_clusters)
    # raise Exception()
    return final_clusters

def merge_invalid_clusters1(clusters, dist_matrix, entity_id2idx_dict, entity_id2k_dict):
    max_k, valid_cluster_ids, invalid_cluster_ids = get_valid_and_invalid_clusters_ids(clusters, entity_id2k_dict)

    all_clusters = [cluster.copy() for cluster in clusters]
    all_cluster_ids = set(range(len(clusters)))

    sorted_clusters_dist = SortedList()
    num_invalid_clusters = len(invalid_cluster_ids)

    logger.info("finding nearest clusters for {}/{} invalid ones".format(len(invalid_cluster_ids), len(all_clusters)))

    for idx, cid1 in enumerate(invalid_cluster_ids):
        # logger.info("found nearest cluster for {}/{}".format(idx, len(invalid_clusters_ids)))
        cluster1 = all_clusters[cid1]
        min_dist = sys.maxsize
        min_cluster_id = None

        for cid2, cluster2 in enumerate(all_clusters):
            # cluster2 = clusters[cidx2]
            if cid2 <= cid1:
                continue

            dist = calculate_distance_between_clusters(cluster1, cluster2, dist_matrix, entity_id2idx_dict)

            if dist < min_dist:
                min_dist = dist
                min_cluster_id = cid2


        sorted_clusters_dist.add((dist, cid1, min_cluster_id))

    new_invalid_clusters_ids = set(invalid_cluster_ids)
    # new_invalid_clusters_ids.extend(invalid_clusters_ids)

    logger.debug("sorted clusters dist: {}".format(sorted_clusters_dist))
    availability = [True for _ in all_clusters]

    logger.info("merging invalid clusters for max_k={}".format(max_k))
    while len(new_invalid_clusters_ids) > 0:
        logger.debug("new_clusters: {}".format(all_clusters))
        # logger.debug("new_valid_clusters_ids: {}".format(new_valid_clusters_ids))
        logger.debug("new_invalid_clusters_ids: {}".format(new_invalid_clusters_ids))
        # logger.debug(sorted_clusters_dist)
        # logger.debug("availability: {}".format(availability))
        logger.debug("num_invalid_clusters: {}".format(num_invalid_clusters))

        smallest_dist, cid1, cid2 = sorted_clusters_dist.pop(0)

        # logger.debug("smallest_dist: {} - cidx1: {} - cidx2: {}".format(smallest_dist, cidx1, cidx2))

        if availability[cid1] and availability[cid2]:
            c1 = all_clusters[cid1]
            c2 = all_clusters[cid2]

            cluster = Cluster.from_iter(itertools.chain(c1, c2))
            logger.debug("c1({}): {} - c2({}): {} - c: {}".format(cid1, c1, cid2, c2, cluster))

            availability[cid1] = False
            availability[cid2] = False

            if cid1 in new_invalid_clusters_ids:
                new_invalid_clusters_ids.remove(cid1)

            if cid2 in new_invalid_clusters_ids:
                new_invalid_clusters_ids.remove(cid2)


            all_clusters.append(cluster)
            availability.append(True)
            new_cluster_idx = len(all_clusters) - 1

            logger.debug("removing invalid dists")
            idx2 = 0
            selected_cluster_ids = [cid1, cid2]
            n_removed_dists = 0
            while idx2 < len(sorted_clusters_dist):
                _, cid1_new, cid2_new = sorted_clusters_dist[idx2]
                if cid1_new in selected_cluster_ids or cid2_new in selected_cluster_ids:
                    sorted_clusters_dist.pop(idx2)
                    n_removed_dists += 1
                idx2 += 1
            logger.debug("removed {}".format(n_removed_dists))

            logger.debug("updating distances")
            is_new_cluster_invalid = len(cluster) < max_k
            if is_new_cluster_invalid:
                new_invalid_clusters_ids.add(new_cluster_idx)
            n_added_dists = 0
            for idx2 in range(len(all_clusters) - 1):
                if availability[idx2]:
                    logger.debug("added distance: {}".format((dist, idx2, new_cluster_idx)))
                    dist = calculate_distance_between_clusters(all_clusters[new_cluster_idx], all_clusters[idx2], dist_matrix, entity_id2idx_dict)

                    sorted_clusters_dist.add((dist, new_cluster_idx, idx2))
                    n_added_dists += 1


            logger.info("remaining invalid clusters: {} - {} (added: {} - removed: {})".format(len(new_invalid_clusters_ids), len(sorted_clusters_dist), n_added_dists, n_removed_dists))


    new_available_clusters = Clusters()
    for cid, cluster in enumerate(all_clusters):
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

