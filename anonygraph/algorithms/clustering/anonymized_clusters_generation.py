import enum
from scipy import stats
import sys

from anonygraph.algorithms import Cluster, Clusters, clustering as calgo
from anonygraph.algorithms.clustering import enforcers
import logging

from sortedcontainers import SortedKeyList
import numpy as np

logger = logging.getLogger(__name__)

class AnonymizedClustersGeneration:
    def __init__(self, enforcer_name, args):
        self.enforcer = enforcers.get_enforcer(enforcer_name, args)

    def run(self, clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict, args):
        new_clusters = self.enforcer.call(clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict)

        return new_clusters


        # deal with small clusters
            # break small clusters and find new clusters for users in broke clusters
                # users that have too high value of k
                    # find users that have too high value of k comparing to remaining users in same cluster
                        # find outliers by using z-score of k
                #

        # print_size_stats(clusters)

        # logger.debug("num invalid clusters: {}".format(len(find_invalid_clusters(clusters, entity_id2k_dict))))

        # num_entities = sum(map(lambda cluster: cluster.num_entities, clusters))
        # removed_entity_ids = split_small_clusters(clusters, entity_id2k_dict)
        # num_entities_after = sum(map(lambda cluster: cluster.num_entities, clusters))

        # logger.debug("num invalid clusters: {}".format(len(find_invalid_clusters(clusters, entity_id2k_dict))))
        # print_size_stats(clusters)

        # logger.debug("num entities before: {} - after: {} ({} + {})".format(num_entities, len(removed_entity_ids)+ num_entities_after, num_entities_after, len(removed_entity_ids)))

        # # deal with too big clusters
        # merge_clusters(clusters, removed_entity_ids, max_dist, dist_matrix, entity_id2idx_dict, entity_id2k_dict)

        # # logger.debug("clusters: {}".format(clusters))
        # logger.debug("num invalid clusters after merging: {}".format(len(find_invalid_clusters(clusters, entity_id2k_dict))))

        # print_size_stats(clusters)

        # split_big_clusters(clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict)

        # raise Exception()

def print_size_stats(clusters):
    sizes = [cluster.num_entities for cluster in clusters]
    max_size = max(sizes)
    min_size = min(sizes)
    mean_size = np.mean(sizes)
    logger.debug("size: max: {} - mean: {} - min: {}".format(max_size, mean_size, min_size))

def calculate_anonymity(clusters, entity_id2k_dict):
    pass

def find_invalid_clusters(clusters, entity_id2k_dict):
    invalid_clusters = []

    for cluster in clusters:
        max_k = max([entity_id2k_dict[entity_id] for entity_id in cluster])

        if cluster.num_entities < max_k:
            invalid_clusters.append(cluster)

    return invalid_clusters

def calculate_real_max_dist(max_dist, dist_matrix):
    # logger.debug(dist_matrix)
    max_pair_dist = np.max(dist_matrix)
    np.fill_diagonal(dist_matrix, max_pair_dist)
    min_pair_dist = np.min(dist_matrix)

    real_max_dist = (max_pair_dist - min_pair_dist) * max_dist + min_pair_dist

    return real_max_dist

def split_small_clusters(clusters, entity_id2k_dict):
    removed_entity_ids = set()

    cluster_id = 0
    while(cluster_id < len(clusters)):
        cluster = clusters[cluster_id]
        # find max k
        # logger.debug("old cluster: {}".format(cluster))

        id2k_values = SortedKeyList(key=lambda item: -item[0])

        for entity_id in cluster:
            entity_k = entity_id2k_dict[entity_id]
            id2k_values.add((entity_k, entity_id))

        while len(id2k_values) > 0 and len(id2k_values) < id2k_values[0][0]:
            entity_id = id2k_values[0][1]

            id2k_values.pop(0)
            cluster.remove_entity(entity_id)
            removed_entity_ids.add(entity_id)

            # logger.debug("current cluster: {}".format(cluster))

        if cluster.num_entities == 0:
            clusters.pop(cluster_id)
        else:
            cluster_id += 1

    return removed_entity_ids


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


def calculate_distance_from_entity_to_cluster(entity_id, cluster, dist_matrix, entity_id2idx_dict):
    entity_idx = entity_id2idx_dict[entity_id]
    cluster_entity_idxes = [entity_id2idx_dict[cluster_entity_id] for cluster_entity_id in cluster]

    distances = dist_matrix[entity_idx, cluster_entity_idxes]
    distance = max(distances)
    return distance


def find_closest_cluster(entity_id, clusters, max_dist, dist_matrix, entity_id2idx_dict, entity_id2k_dict):
    entity_k = entity_id2k_dict[entity_id]

    closest_cluster = None
    smallest_dist = sys.maxsize
    for cluster in clusters:
        distance = calculate_distance_from_entity_to_cluster(entity_id, cluster, dist_matrix, entity_id2idx_dict)
        k_values = [entity_id2k_dict[entity_id] for entity_id in cluster] + [entity_k]
        max_k = max(k_values)

        if distance > max_dist or cluster.num_entities + 1 >= max_k * 2:
            continue

        if distance < smallest_dist:
            smallest_dist = distance
            closest_cluster = cluster

    return closest_cluster

def split_big_clusters(clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
    index = 0

    count = 0
    while(index < len(clusters)):
        logger.info("{}/{} clusters".format(index + 1, len(clusters)))
        current_cluster = clusters.pop(index)
        logger.info("{}/{} clusters".format(index + 1, len(clusters)))

        max_k = max([entity_id2k_dict[entity_id] for entity_id in current_cluster])

        if len(current_cluster) >= max_k * 2:
            new_clusters_list = split_big_cluster(current_cluster, max_k, dist_matrix, entity_id2idx_dict, entity_idx2id_dict)

            for new_cluster in new_clusters_list:
                logger.info(new_cluster)

                # raise Exception()
                clusters.insert(index, new_cluster)

            logger.info("break cluster of size {} into {}".format(len(current_cluster), len(new_clusters_list)))

            # if count == 1:
            #     raise Exception()
            # else:
            #     count += 1
        else:
            clusters.insert(index, current_cluster)
            index += 1
            logger.info("add cluster: {}".format(current_cluster))

        logger.info("{}/{} clusters".format(index + 1, len(clusters)))

def generate_distance_matrix(entity_idxes, dist_matrix):
    result = np.zeros((len(entity_idxes), len(entity_idxes)))

    for entity1_idx, raw_entity1_idx in enumerate(entity_idxes):
        for entity2_idx, raw_entity2_idx in enumerate(entity_idxes):
            result[entity1_idx, entity2_idx] = dist_matrix[raw_entity1_idx, raw_entity2_idx]

    return result

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

