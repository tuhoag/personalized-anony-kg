import numpy as np
import logging

from anonygraph import algorithms
import anonygraph
from anonygraph.algorithms.clustering import clustering_algorithms as calgo
from .outliers_removal import OutlierRemoval
import anonygraph.algorithms.clustering.outliers_handlers as ohandlers
from anonygraph.constants import *


logger = logging.getLogger(__name__)

class RawClustersGeneration:
    def __init__(self, calgo_name, outliers_handler_name, args):
        self.calgo_fn = calgo.get_clustering_algo(calgo_name)
        self.outliers_handler_fn = ohandlers.get_outliers_handler(outliers_handler_name, args)

    def run(self, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict, args):
        remaining_entity_ids = self.outliers_handler_fn(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict)
        remaining_entity_idxes = list(map(lambda entity_id: entity_id2idx_dict[entity_id], remaining_entity_ids))
        new_dist_matrix = generate_distance_matrix(remaining_entity_idxes, dist_matrix)

        new_idx2id_dict = {}
        new_id2idx_dict = {}
        new_id2k_dict = {}

        for new_entity_idx, entity_idx in enumerate(remaining_entity_idxes):
            entity_id = entity_idx2id_dict[entity_idx]

            new_idx2id_dict[new_entity_idx] = entity_id
            new_id2idx_dict[entity_id] = new_entity_idx
            new_id2k_dict[entity_id] = entity_id2k_dict[entity_id]


        idx_clusters = self.calgo_fn(new_dist_matrix, new_id2idx_dict, new_idx2id_dict, new_id2k_dict, args)

        return convert_idx2id_clusters(idx_clusters, new_idx2id_dict)

def convert_idx2id_clusters(clusters, entity_idx2id_dict):
    new_clusters = algorithms.Clusters()

    for cluster in clusters:
        new_cluster = algorithms.Cluster()

        for entity_idx in cluster:
            new_cluster.add_entity(entity_idx2id_dict[entity_idx])

        new_clusters.add_cluster(new_cluster)

    return new_clusters


def generate_distance_matrix(entity_idxes, dist_matrix):
    result = np.zeros((len(entity_idxes), len(entity_idxes)))

    for entity1_idx, raw_entity1_idx in enumerate(entity_idxes):
        for entity2_idx, raw_entity2_idx in enumerate(entity_idxes):
            result[entity1_idx, entity2_idx] = dist_matrix[raw_entity1_idx, raw_entity2_idx]

    return result