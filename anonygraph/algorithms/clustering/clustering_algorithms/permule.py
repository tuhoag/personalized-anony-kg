import logging
import math
from anonygraph.algorithms.cluster import Cluster
import numpy as np

import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

class PermuleAlgorithm:
    def __init__(self):
        pass

    def run(self, dist_matrix, id2idx_dict, idx2id_dict, id2k_dict):
        clusters = algo.Clusters()
        num_entities = len(id2idx_dict)

        remaining_entity_ids = list(range(num_entities))

        while len(remaining_entity_ids) > 0:
            entity1_idx = remaining_entity_ids.pop(0)

            cluster = Cluster()
            cluster.add_entity(entity_id=entity1_idx)

            idx = 0
            while(len(remaining_entity_ids) > 0):
                if dist_matrix[entity1_idx, idx] == 0:
                    entity2_idx = remaining_entity_ids.pop(idx)
                    cluster.add_entity(entity_id=entity2_idx)
                else:
                    idx += 1

        return clusters