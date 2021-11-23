import math
import matplotlib.pyplot as plt
import enum
import numpy as np
from sortedcontainers import SortedDict
import logging

logger = logging.getLogger(__name__)

class RatioRandomNormalGenerator:
    def __init__(self, max_k_ratio, num_k_values=10):
        self.max_k_ratio = max_k_ratio
        self.num_k_values = num_k_values

        logger.debug("max k ratio: {} - num k vals: {}".format(self.max_k_ratio, self.num_k_values))

    def __call__(self, graph):
        entity_ids = list(graph.entity_ids)
        num_entities = len(entity_ids)

        max_k = int(num_entities * self.max_k_ratio)

        if self.num_k_values == 1:
            k_values = np.array([max_k])
        else:
            min_k = 1
            k_step = max(math.ceil((max_k - min_k) / self.num_k_values), 1)
            logger.debug("k settings: max: {} - min: {} - step: {}".format(max_k, min_k, k_step))
            raw_k_values = np.arange(min_k, max_k + 1, k_step)
            logger.debug("raw generated: {}".format(raw_k_values))
            k_values = np.arange(min_k, max_k + 1, k_step)[:self.num_k_values]

        logger.debug("generated k vals: {}".format(k_values))
        assert len(k_values) <= self.num_k_values, "generated {} ({} vals) while expected {} vals".format(k_values, len(k_values), self.num_k_values)

        random_numbers = np.random.normal(size=len(entity_ids))
        min_random_val = min(random_numbers)
        max_random_val = max(random_numbers)
        random_k_idxes = ((random_numbers - min_random_val) / (max_random_val - min_random_val) * (len(k_values)-1)).astype(int)
        logger.debug("random k idxes: {}".format(random_k_idxes))
        random_k_vals = np.vectorize(convert_id2val(k_values))(random_k_idxes)

        # show_hist(random_k_vals)

        entity_id2k_dict = {}

        for entity_idx, k in enumerate(random_k_vals):
            entity_id2k_dict[entity_ids[entity_idx]] = k

        return entity_id2k_dict


def print_count_dict(numbers):
    count_dict = SortedDict()

    for number in numbers:
        count = count_dict.get(number, 0)
        count += 1
        count_dict[number] = count

    # temp_list =
    logger.debug(count_dict)

def show_hist(vals):
    val_dict = {}
    for val in vals:
        count = val_dict.get(val, 0)
        count += 1
        val_dict[val] = count

    counts = []
    for val in vals:
        counts.append(val_dict[val])

    plt.bar(x=vals, height=counts)
    plt.show()


def convert_id2val(list_vals):
    def _convert(x):
        return list_vals[x]
    return _convert
