from anonygraph.algorithms import clustering
from .base_enforcer import BaseEnforcer
from anonygraph.algorithms import Clusters
from anonygraph.constants import *

class SmallRemovalEnforcer(BaseEnforcer):
    def __init__(self, args):
        super().__init__(SR_ENFORCER, args)

    def call(self, clusters, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
        new_clusters = Clusters()

        for cluster in clusters:
            max_k = 0

            for entity_id in cluster:
                entity_k = entity_id2k_dict[entity_id]
                max_k = max(max_k, entity_k)

            if cluster.num_entities >= max_k:
                new_clusters.add_cluster(cluster.copy())

        return new_clusters

