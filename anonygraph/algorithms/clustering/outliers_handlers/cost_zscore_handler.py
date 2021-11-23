import logging

import numpy as np
from scipy import stats
import heapq

from anonygraph.constants import *
from .base_handler import BaseHandler

logger = logging.getLogger(__name__)

class CostZscoreHandler(BaseHandler):
    def __init__(self, max_cost):
        super().__init__(COST_ZSCORE_HANDLER)
        self.__max_cost = max_cost

    def __call__(self, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
        entity_cost = calculate_anonymization_cost(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict)

        logger.debug(entity_cost)
        zscore_cost = stats.zscore(entity_cost)

        logger.debug("zscore: {}".format(zscore_cost))

        remaining_entity_ids = []
        for entity_idx in range(zscore_cost.shape[0]):
            logger.debug("{} - {}".format(entity_idx, zscore_cost[entity_idx]))
            if zscore_cost[entity_idx] <= self.__max_cost:
                remaining_entity_ids.append(entity_idx2id_dict[entity_idx])

        logger.debug("remaining ids: {}".format(remaining_entity_ids))

        return remaining_entity_ids

def calculate_anonymization_cost(dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
    entity_cost = np.zeros(dist_matrix.shape[0])

    for entity_id, entity_k in entity_id2k_dict.items():
        # find k closest entities
        logger.debug("entity id: {}".format(entity_id))

        entity_idx = entity_id2idx_dict[entity_id]
        kclosest_costs2 = heapq.nsmallest(entity_k, dist_matrix[entity_idx,:])
        kclosest_costs = np.partition(dist_matrix[entity_idx,:], entity_k - 1)[:entity_k]

        logger.debug("{} closest cost: {} (all {})".format(entity_k, kclosest_costs, dist_matrix[entity_idx,:]))
        logger.debug("{} closest closts 2: {}".format(entity_k, kclosest_costs2))

        # calculate sum distance
        sum_dists = np.sum(kclosest_costs)
        sum_dists2 = np.sum(kclosest_costs2)

        assert abs(sum_dists - sum_dists2) < 0.0000000001, "Calculated from partition: {} != that of nsmallest: {} (diff: {})".format(sum_dists, sum_dists2, abs(sum_dists - sum_dists2))
        logger.debug("sum dists: {}".format(sum_dists))
        logger.debug("sum dists 2: {}".format(sum_dists2))

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