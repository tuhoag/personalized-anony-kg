import logging
import math
import numpy as np

from sklearn_extra.cluster import KMedoids

import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

class KMedoidsAlgorithm:
    def __init__(self, num_clusters):
        self.__num_clusters = num_clusters

    def run(self, dist_matrix):
        if self.__num_clusters > 1:
            # logger.debug("num clusters: {} ({}/{})".format(num_clusters, len(entity_ids), self.__min_size))
            algo_fn = KMedoids(n_clusters=self.__num_clusters, init="k-medoids++", metric="precomputed")
            sk_clusters = algo_fn.fit_predict(dist_matrix)
            logger.debug(sk_clusters)

            clusters = convert_sklearn_clustering_results_to_cluster(sk_clusters)
        else:
            clusters = [algo.Cluster.from_iter(range(dist_matrix.shape[0]))]

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