import logging
import math
import numpy as np
import hdbscan

import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

class HDBSCANAlgorithm:
    def __init__(self, min_cluster_size):
        self.__min_cluster_size = int(min_cluster_size)

    def run(self, dist_matrix):
        algo = hdbscan.HDBSCAN(min_cluster_size=self.__min_cluster_size, metric="precomputed")
        algo.fit(dist_matrix)

        users_clusters = algo.labels_

        clusters = convert_sklearn_clustering_results_to_cluster(users_clusters)

        return clusters

def convert_sklearn_clustering_results_to_cluster(clustering_results):
    results_dict = {}

    for entity_idx, cluster_id in enumerate(clustering_results):
        cluster = results_dict.get(cluster_id)

        if cluster is None:
            cluster = algo.Cluster()
            results_dict[cluster_id] = cluster

        cluster.add_entity(entity_idx)

    return list(results_dict.values())