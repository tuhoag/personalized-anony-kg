import matplotlib.pyplot as plt
import enum
import numpy as np
from sortedcontainers import SortedDict
import logging

logger = logging.getLogger(__name__)

class RandomNormalGenerator:
    def __init__(self, k_range):
        self.k_values = list(k_range)

    def __call__(self, graph):
        entity_ids = list(graph.entity_ids)

        random_numbers = np.random.normal(size=len(entity_ids))
        min_random_val = min(random_numbers)
        max_random_val = max(random_numbers)
        random_k_idxes = ((random_numbers - min_random_val) / (max_random_val - min_random_val) * (len(self.k_values)-1)).astype(int)

        random_k_vals = np.vectorize(convert_id2val(self.k_values))(random_k_idxes)

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
