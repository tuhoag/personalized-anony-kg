import math
import matplotlib.pyplot as plt
import enum
import numpy as np
from scipy.stats import zipf
from scipy.stats.stats import moment
from sortedcontainers import SortedDict
import logging

from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class RangeZipfGenerator(BaseGenerator):
    def __init__(self, min_k, max_k, step_k, param=1.5):
        self.max_k = max_k
        self.min_k = min_k
        self.step_k = step_k
        self.param = param

        if self.min_k > self.max_k:
            raise Exception("min k ({}) should be less than max k ({})".format(self.min_k, self.max_k))

        logger.debug("({}, {}, {}) - param: {}".format(self.min_k, self.max_k, self.step_k, self.param))

    def generate_k_values(self, entity_ids, graph):
        num_entities = len(entity_ids)

        # generate k values
        max_k = self.max_k + 1
        min_k = self.min_k
        step_k = self.step_k

        k_values = np.arange(min_k, max_k, step_k)

        logger.debug("({}, {}, {}) - k values: {}".format(min_k, max_k, step_k, k_values))

        # generate random numbers
        random_numbers = (np.random.zipf(a=self.param, size=num_entities) - 1).tolist()
        # random_numbers = zipf.stats(self.param, moments="mvsk")
        logger.debug(random_numbers)

        # logger.debug("param {} - ratio param: {}".format(self.param, ratio_param))

        random_k_vals = []
        max_k_index = len(k_values) - 1
        index_list = list(range(len(k_values)))

        for i in range(len(entity_ids)):
            if random_numbers[i] > max_k_index:
                k_index = np.random.choice(index_list)
                # k_index = 0
            else:
                k_index = random_numbers[i]

            k_value = k_values[k_index]
            random_k_vals.append(k_value)

        logger.debug(random_k_vals)
        # show_hist(random_k_vals)

        return random_k_vals


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
    plt.xticks(vals)
    plt.show()
