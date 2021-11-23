from anonygraph.algorithms import cluster
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import heapq
import logging
import math
import numpy as np
from sortedcontainers import SortedKeyList
import itertools

from sklearn_extra.cluster import KMedoids

import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

class PSizeAlgorithm:
    def run(self, dist_matrix, id2idx_dict, idx2id_dict, id2k_dict):
        # initialize id2cost_dict
        k_seq = initialize_k_sequence(id2idx_dict, id2k_dict)
        logger.debug("k_seq: {}".format(k_seq))

        # calculate d_core_seq
        dnearest_seq = calculate_idx2dnearest_seq(dist_matrix, k_seq)
        logger.debug("dnearest_seq: {}".format(dnearest_seq))

        # calculate d2 distance
        new_dist_matrix = calculate_dnearest_dist_matrix(dist_matrix, dnearest_seq, k_seq)


        d2nearest_seq = calculate_idx2dnearest_seq(new_dist_matrix, k_seq)

        logger.debug("d2nearst_seq: {}".format(d2nearest_seq))

        remaining_idxes = list(range(len(dnearest_seq)))

        clusters = algo.Clusters()

        while(len(remaining_idxes) > 0):
            current_idx = np.argmin(d2nearest_seq)

            logger.debug("remaining {} entities".format(len(remaining_idxes)))
            cluster = find_valid_clusters(current_idx, new_dist_matrix, k_seq, remaining_idxes)

            logger.debug("found a cluster: {} for user: {}".format(cluster, idx2id_dict[current_idx]))
            clusters.add_cluster(cluster)

            for idx in cluster:
                d2nearest_seq[idx] = sys.maxsize
        # generate clusters of idxes

        return clusters

def find_valid_clusters(idx, dist_matrix, k_seq, remaining_idxes):
    cluster_list = [idx]

    logger.debug("{} - remaining: {}".format(idx, len(remaining_idxes)))
    remaining_idxes.remove(idx)

    # find smallest
    required_anonymity = k_seq[idx]

    while len(cluster_list) < required_anonymity and len(remaining_idxes) > 0:
        best_idx = find_best_entity_idx(dist_matrix, cluster_list, remaining_idxes)

        cluster_list.append(best_idx)
        required_anonymity = max(required_anonymity, k_seq[best_idx])

        remaining_idxes.remove(best_idx)


    cluster = algo.Cluster.from_iter(cluster_list)

    return cluster

    # return cluster


def find_best_entity_idx(dist_matrix, cluster_list, remaining_idxes):
    smallest_idx = -1
    smallest_score = sys.maxsize

    logger.debug("dist matrix shape: {}".format(dist_matrix.shape))
    logger.debug("remaining idxes: {}".format(remaining_idxes))
    logger.debug("cluster list: {}".format(cluster_list))
    # logger.debug(dist_matrix[remaining_idxes, cluster_list].shape)

    # cluster_dist = np.max(dist_matrix[remaining_idxes, cluster_list], axis=1, keepdims=True)
    # logger.debug(cluster_dist)

    for idx in remaining_idxes:
        if idx in cluster_list:
            continue

        dist = max(dist_matrix[idx, cluster_list])

        if dist < smallest_score:
            smallest_score = dist
            smallest_idx = idx

    return smallest_idx


def visualize_norm_seq(dnearest_seq, k_seq):
    max_dnearest = max(dnearest_seq)
    norm_dnearest_seq = dnearest_seq / max_dnearest


    plt.plot(norm_dnearest_seq, k_seq, 'o')
    plt.xlabel("cost")
    plt.ylabel("k")
    plt.show()

def visualize(dnearest_seq, k_seq, new_dist_matrix):
    visualize_norm_seq(dnearest_seq, k_seq)

    d2nearest_seq = calculate_idx2dnearest_seq(new_dist_matrix, k_seq)
    visualize_norm_seq(d2nearest_seq, k_seq)

def calculate_dnearest_dist_matrix(dist_matrix, dnearest_seq, k_seq):
    new_dist_matrix = np.zeros_like(dist_matrix)
    num_entities = dist_matrix.shape[0]

    # logger.debug(dnearest_seq)

    for idx1, idx2 in itertools.combinations(range(num_entities), r=2):
        nearest_dist = max(dnearest_seq[idx1], dnearest_seq[idx2], dist_matrix[idx1, idx2])

        # logger.debug("nearest_dist: {}".format(nearest_dist))
        # raise Exception()
        max_k = max(k_seq[idx1], k_seq[idx2])

        knearest_dist = nearest_dist * max_k
        new_dist_matrix[idx1, idx2] = knearest_dist
        new_dist_matrix[idx2, idx1] = knearest_dist

    return new_dist_matrix

def find_knearest_unremoved_idxes(closest_idxes, remaining_idxes_set, k):
    knearest_idxes = []
    cost = 0
    count = 0

    logger.debug("knearest idxes: {}".format(knearest_idxes))

    for idx2, dist in closest_idxes:
        logger.debug("idx: {} - dist: {}".format(idx2, dist))

        if idx2 in remaining_idxes_set:
            knearest_idxes.append(idx2)
            count += 1
            cost += dist

        logger.debug("knearest idxes: {} - cost: {}".format(knearest_idxes, cost))

        if count == k:
            break

    return knearest_idxes, cost


def initialize_idx2cost_list(remaining_idxes_set, idx2k_dict, idx2closest_idx_dict):
    logger.debug("idx2k: {}".format(idx2k_dict))
    logger.debug("remaining idxes: {}".format(remaining_idxes_set))

    result = SortedKeyList(key=lambda item: item[1])

    for idx in remaining_idxes_set:
        logger.debug("idx: {}".format(idx))

        k = idx2k_dict[idx]
        closest_idxes = idx2closest_idx_dict[idx]

        logger.debug("k: {} - closest idxes: {}".format(k, closest_idxes))

        kclosest_idxes, cost = find_knearest_unremoved_idxes(closest_idxes, remaining_idxes_set, k)

        logger.debug("{} closest idxes: {} - cost: {}".format(k, kclosest_idxes, cost))

        result.add((idx, cost))

    return result

def initialize_idx2k_dict(id2idx_dict, id2k_dict):
    idx2k_dict = {}

    for id, idx in id2idx_dict.items():
        idx2k_dict[idx] = id2k_dict[id]

    return idx2k_dict

def initialize_k_sequence(id2idx_dict, id2k_dict):
    k_sequence = np.zeros(len(id2idx_dict), dtype=int)

    for id, idx in id2idx_dict.items():
        k_sequence[idx] = id2k_dict[id]

    return k_sequence

def calculate_idx2dnearest_seq(dist_matrix, k_sequence):
    num_entities = dist_matrix.shape[0]
    idx2dnearest_seq = np.zeros(num_entities)

    for idx in range(num_entities):
        k = k_sequence[idx]
        # k = 5

        logger.debug("idx: {} - dist: {}".format(idx, dist_matrix[idx, :]))
        # k_smallest_vals = np.partition(dist_matrix[idx, :], k)[:k]
        k_smallest_vals = heapq.nsmallest(k, dist_matrix[idx, :])

        # logger.debug("smallest: {}".format(k_smallest_vals))
        # raise Exception()
        idx2dnearest_seq[idx] = np.max(k_smallest_vals)


        logger.debug("{} smallest: {} - max: {}".format(k, k_smallest_vals, idx2dnearest_seq[idx]))
        # logger.debug("{} smallest: {} - max: {}".format(k, k_smallest_vals2, np.max(k_smallest_vals2)))

        # raise Exception()
    return idx2dnearest_seq

def initialize_idx2closest_idx_dict(dist_matrix, idx2k_dict):
    idx2closest_idx_dict = {}

    for idx, k in idx2k_dict.items():
        # find k closest idxes
        closest_idxes = SortedKeyList(key=lambda item: item[1])

        for idx2 in range(len(idx2k_dict)):
            dist = dist_matrix[idx, idx2]
            closest_idxes.add((idx2, dist))

        idx2closest_idx_dict[idx] = closest_idxes

    return idx2closest_idx_dict



def convert_sklearn_clustering_results_to_cluster(clustering_results):
    results_dict = {}

    for entity_idx, cluster_id in enumerate(clustering_results):
        cluster = results_dict.get(cluster_id)

        if cluster is None:
            cluster = algo.Cluster()
            results_dict[cluster_id] = cluster

        cluster.add_entity(entity_idx)

    return list(results_dict.values())