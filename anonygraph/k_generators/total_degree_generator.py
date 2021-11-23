import math
import matplotlib.pyplot as plt
import enum
import numpy as np
from sortedcontainers import SortedDict
import logging

from anonygraph.info_loss import info
from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)

class TotalDegreeGenerator(BaseGenerator):
    def __init__(self, min_k_ratio, max_k_ratio, step_k_ratio):
        self.max_k_ratio = max_k_ratio
        self.min_k_ratio = min_k_ratio
        self.step_k_ratio = step_k_ratio

        if self.min_k_ratio > self.max_k_ratio:
            raise Exception("min ratio ({}) should be less than max ratio ({})".format(self.min_k_ratio, self.max_k_ratio))

        logger.debug("({}, {}, {})".format(self.min_k_ratio, self.max_k_ratio, self.step_k_ratio))

    def generate_k_values(self, entity_ids, graph):
        num_entities = len(entity_ids)

        # generate k values
        max_k = int(num_entities * self.max_k_ratio)
        min_k = int(num_entities * self.min_k_ratio)
        step_k = int(num_entities * self.step_k_ratio)

        k_values = np.arange(min_k, max_k, step_k)

        logger.debug("({}, {}, {}) - k values: {}".format(min_k, max_k, step_k, k_values))

        # sorted users based on their total degree
        #
        total_degrees = generate_total_degree_sequence(entity_ids, graph)
        # logger.debug(sorted(total_degrees))

        max_degree = max(total_degrees)
        min_degree = min(total_degrees)

        logger.debug("sorted k values: {}".format(k_values))
        random_k_num = (total_degrees - min_degree) / (max_degree - min_degree) * (len(k_values) - 1)
        random_k_idxes = random_k_num.astype(int)

        # idx = np.argmax(total_degrees)
        # logger.debug("max: {} - min: {}".format(max_degree, min_degree))
        # logger.debug("degree: {} - num: {} index: {}".format(total_degrees[idx], random_k_num[idx], random_k_idxes[idx]))
        # logger.debug("max idx: {} - min idx: {}".format(max(random_k_idxes), min(random_k_idxes)))
        # logger.debug(random_k_index_sequence[-1])

        random_k_vals = np.vectorize(convert_id2val(k_values))(random_k_idxes)

        visualize_k_vals(random_k_vals, total_degrees, random_k_idxes)

        return random_k_vals

def generate_total_degree_sequence(entity_ids, graph):
    degree_sequence = np.zeros(len(entity_ids))

    for idx, entity_id in enumerate(entity_ids):
        # calculate out-degree
        out_degree_info = info.get_degree_info(graph, entity_id, "out")
        in_degree_info = info.get_degree_info(graph, entity_id, "in")
        attrs_info = info.get_attribute_info(graph, entity_id)

        out_degree = sum(out_degree_info.values())
        in_degree = sum(in_degree_info.values())
        attrs_degree = sum(map(lambda attr_vals: len(attr_vals), attrs_info.values()))

        logger.debug("id: {} - attr degree: {} (info: {}".format(entity_id, attrs_degree, attrs_info))
        logger.debug("id: {} - in degree: {} (info: {}".format(entity_id, in_degree, in_degree_info))
        logger.debug("id: {} - out degree: {} (info: {})".format(entity_id, out_degree, out_degree_info))

        # raise Exception()
        degree_sequence[idx] = out_degree + in_degree + attrs_degree

    return degree_sequence

def get_count_dict(numbers):
    count_dict = SortedDict()

    for number in numbers:
        count = count_dict.get(number, 0)
        count += 1
        count_dict[number] = count

    # temp_list =
    return count_dict


def visualize_k_vals(k_vals, degrees, idxes):
    max_degree = np.max(degrees)
    min_degree = 0
    step_degree = 10
    degree_levels = np.arange(min_degree, max_degree, step_degree)
    degree_count_dict = get_count_dict(degrees)
    logger.debug(degree_count_dict)

    k_count_dict = get_count_dict(k_vals)
    logger.debug(k_count_dict)

    idx_count_dict = get_count_dict(idxes)
    logger.debug(idx_count_dict)

    plt.hist(degrees, bins=len(k_count_dict))
    # plt.yticks(degree_levels)
    # plt.hist(k_vals)
    plt.show()

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
