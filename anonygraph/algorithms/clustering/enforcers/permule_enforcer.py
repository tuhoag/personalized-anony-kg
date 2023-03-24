import itertools
import logging
import time
import numpy as np
import sys
from sortedcontainers import SortedList
from joblib import Parallel, delayed

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
        self.num_workers = int(args[0])

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
        new_clusters = merge_invalid_clusters(clusters, new_dist_matrix, entity_id2idx_dict, entity_id2k_dict, self.num_workers)

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

def initialize_sorted_dist(all_clusters, c_in_id, invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict, cache_dist):
    sorted_dist = SortedList()

    for c_id in available_cluster_ids:
        if c_id == c_in_id:
            continue

        if c_in_id < c_id:
            key = (c_in_id, c_id)
        else:
            key = (c_id, c_in_id)

        dist = cache_dist.get(key)
        if key is None:
            dist = calculate_distance_between_clusters(all_clusters[c_in_id], all_clusters[c_id], dist_matrix, entity_id2idx_dict)
            cache_dist[key] = dist

        sorted_dist.add((dist, c_id))
    return sorted_dist

def initialize_cache_dist(all_clusters, invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict):
    cache_dist2 = {}

    cache_dist = {}
    for c_in_id in invalid_cluster_ids:
        sorted_dist = initialize_sorted_dist(all_clusters, c_in_id, invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict, cache_dist)

        cache_dist2[c_in_id] = sorted_dist

    logger.debug("initialial invalid_cluster_ids: {}".format(invalid_cluster_ids))
    logger.debug("initialial cache_dist2: {}".format(cache_dist2))
    return cache_dist2

def merge_invalid_clusters(clusters, dist_matrix, entity_id2idx_dict, entity_id2k_dict, num_workers):
    max_k, valid_cluster_ids, invalid_cluster_ids = get_valid_and_invalid_clusters_ids(clusters, entity_id2k_dict)

    all_clusters = [cluster.copy() for cluster in clusters]
    available_cluster_ids = set(range(len(clusters)))

    # logger.info("finding nearest clusters for {}/{} invalid ones".format(len(invalid_cluster_ids), len(all_clusters)))
    logger.info("initializing cache dist")
    # cache_dist2 = initialize_cache_dist(all_clusters, invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict)

    # logger.debug(cache_dist2)
    # raise Exception()
    cache_dist = {}
    logger.info("start merging")
    while(len(invalid_cluster_ids) > 0):
        logger.debug("all_clusters: {}".format(all_clusters))
        start_time = time.time()

        # find smallest dist invalid cluster
        # min_dist = sys.maxsize
        # nearest_cluster_ids = None
        # for c_in_id in invalid_cluster_ids:
        #     # logger.debug(invalid_cluster_ids)
        #     # logger.debug(cache_dist2)
        #     c_sorted_dist = cache_dist2[c_in_id]

        #     dist, c_id = c_sorted_dist.pop(0)
        #     while(c_id not in available_cluster_ids):
        #         dist, c_id = c_sorted_dist.pop(0)

        #     if dist < min_dist:
        #         min_dist = dist
        #         nearest_cluster_ids = (c_in_id, c_id)


        min_dist = sys.maxsize
        nearest_cluster_ids = None

        for c_in_id in invalid_cluster_ids:
            for c_id in available_cluster_ids:
                if c_id == c_in_id:
                    continue

                if c_in_id > c_id:
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

        logger.debug("selected dist: {} - pair: {}".format(min_dist, nearest_cluster_ids))

        new_cluster = Cluster.from_iter(itertools.chain(all_clusters[nearest_cluster_ids[0]], all_clusters[nearest_cluster_ids[1]]))
        new_cluster_id = len(all_clusters)

        logger.debug("new_cluster ({}): {}".format(new_cluster_id, new_cluster))
        all_clusters.append(new_cluster)
        available_cluster_ids.add(new_cluster_id)

        available_cluster_ids.remove(nearest_cluster_ids[0])
        available_cluster_ids.remove(nearest_cluster_ids[1])

        if len(all_clusters[nearest_cluster_ids[0]]) < max_k:
            invalid_cluster_ids.remove(nearest_cluster_ids[0])

        if len(all_clusters[nearest_cluster_ids[1]]) < max_k:
            invalid_cluster_ids.remove(nearest_cluster_ids[1])

        # del cache_dist2[nearest_cluster_ids[0]]
        # del cache_dist2[nearest_cluster_ids[1]]

        # for c_in_id, c_sorted_dist in cache_dist2.items():
        #     dist = calculate_distance_between_clusters(all_clusters[c_in_id], all_clusters[new_cluster_id], dist_matrix, entity_id2idx_dict)

        #     c_sorted_dist.add((dist, new_cluster_id))

        if len(new_cluster) < max_k:
            invalid_cluster_ids.add(new_cluster_id)

            # sorted_dist = initialize_sorted_dist(all_clusters, new_cluster_id, invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict)
            # cache_dist2[new_cluster_id] = sorted_dist

        logger.debug("invalid_cluster_ids: {}".format(invalid_cluster_ids))
        # logger.debug("cache_dist2: {}".format(cache_dist2))
        logger.info("num invalid clusters: {} in {}".format(len(invalid_cluster_ids), time.time() - start_time))

    final_clusters = Clusters()
    for c_id in available_cluster_ids:
        final_clusters.add_cluster(all_clusters[c_id])

    logger.debug(final_clusters)
    # raise Exception()
    return final_clusters

def find_min_distance(start, stop, all_clusters, list_invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict, cache_dist):
    min_dist = sys.maxsize
    nearest_cluster_ids = None

    for c_in_idx in range(start, stop + 1):
        c_in_id = list_invalid_cluster_ids[c_in_idx]

        for c_id in available_cluster_ids:
            if c_id == c_in_id:
                continue

            if c_in_id > c_id:
                    key = (c_in_id, c_id)
            else:
                key = (c_id, c_in_id)

            dist = cache_dist.get(key)
            if dist is None:
                dist = calculate_distance_between_clusters(all_clusters[c_in_id], all_clusters[c_id], dist_matrix, entity_id2idx_dict)
                cache_dist[key] = dist

            dist = calculate_distance_between_clusters(all_clusters[c_in_id], all_clusters[c_id], dist_matrix, entity_id2idx_dict)

            if dist < min_dist:
                min_dist = dist
                nearest_cluster_ids = (c_in_id, c_id)

    return min_dist, nearest_cluster_ids, cache_dist

def calculate_min_distance_parallel(all_clusters, list_invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict, cache_dist, num_workers):
    size = int(len(list_invalid_cluster_ids) / num_workers)
    sub_invalid_cluster_ids = []
    start = 0
    stop = size
    for i in range(num_workers - 1):
        stop = min(start + size - 1, len(list_invalid_cluster_ids))
        sub_invalid_cluster_ids.append((start, stop))

        start = stop + 1

    sub_invalid_cluster_ids.append((start, len(list_invalid_cluster_ids) - 1))

    logger.debug(sub_invalid_cluster_ids)

    results = Parallel(n_jobs=8)(delayed(find_min_distance)(start, stop, all_clusters, list_invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict, cache_dist) for (start, stop) in sub_invalid_cluster_ids)

    logger.debug(results)

    min_dist = sys.maxsize
    nearest_cluster_ids = None

    for dist, pair, sub_cache_dist in results:
        if dist < min_dist:
            min_dist = dist
            nearest_cluster_ids = pair

        cache_dist.update(sub_cache_dist)

    logger.debug(nearest_cluster_ids)

    return nearest_cluster_ids

def calculate_min_distance(all_clusters, invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict, cache_dist, num_workers):
    if num_workers > 1:
        return calculate_min_distance_parallel(all_clusters, invalid_cluster_ids, available_cluster_ids, dist_matrix, entity_id2idx_dict, cache_dist, num_workers)
    else:
        min_dist = sys.maxsize
        nearest_cluster_ids = None

        for c_in_id in invalid_cluster_ids:
            for c_id in available_cluster_ids:
                if c_id == c_in_id:
                    continue

                if c_in_id > c_id:
                    key = (c_in_id, c_id)
                else:
                    key = (c_id, c_in_id)

                dist = cache_dist.get(key)
                if dist is None:
                    dist = calculate_distance_between_clusters(all_clusters[c_in_id], all_clusters[c_id], dist_matrix, entity_id2idx_dict)

                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster_ids = (c_in_id, c_id)

        return nearest_cluster_ids

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

