import numpy as np

import logging

logger = logging.getLogger(__name__)


class OutlierRemoval:
    def __init__(self):
        pass

    def call(self, max_dist, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict, args):

        entity_cost = calculate_anonymization_cost(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict)

        logger.debug(entity_cost)

        raise Exception()


def calculate_anonymization_cost(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
    entity_cost = np.zeros(dist_matrix.shape[0])

    for entity_id, entity_k in entity_id2k_dict.items():
        # find k closest entities
        entity_idx = entity_id2idx_dict[entity_id]
        kclosest_costs = np.partition(dist_matrix[entity_idx,:], entity_k)

        # calculate sum distance
        sum_dists = np.sum(kclosest_costs)
        logger.debug("sum dists: {}".format(sum_dists))

        entity_cost[entity_idx] = sum_dists

    return entity_cost

def find_closest_entity_ids(entity_id, dist_matrix, num_entities, entity_id2idx_dict, entity_idx2id_dict):
    entity_idx = entity_id2idx_dict[entity_id]
    closest_idxes = np.argpartition(dist_matrix[entity_idx,:], num_entities)

    logger.debug("dists: {}".format(dist_matrix[entity_idx, :]))
    logger.debug("closest idx: {}".format(closest_idxes))

    result = dist_matrix[entity_idx,closest_idxes[:num_entities]]
    closest_ids = set(map(lambda idx: entity_idx2id_dict[idx], closest_idxes[:num_entities]))

    return closest_ids