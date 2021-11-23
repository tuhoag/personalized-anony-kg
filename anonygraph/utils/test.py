import logging
import os

from anonygraph.constants import *
import anonygraph.utils.runner as rutils

logger = logging.getLogger(__name__)

def calculate_num_invalid_clusters(clusters, id2k_dict):
    count = 0

    for cluster in clusters:
        max_k = calculate_max_k(cluster, id2k_dict)
        # logger.debug("cluster: {}".format(cluster))


        if cluster.num_entities < max_k:
            count += 1
            # raise Exception()

    # logger.debug("{}/{} invalid clusters".format(count, len(clusters)))
    # raise Exception()
    return count

def calculate_max_k(cluster, id2k_dict):
    k_vals = list(map(lambda id: id2k_dict[id], cluster))
    max_k = max(k_vals)
    # logger.debug("k vals: {} - max: {}".format(k_vals, max_k))
    return max_k


def calculate_num_big_clusters(clusters, id2k_dict):
    count = 0

    for cluster in clusters:
        max_k = calculate_max_k(cluster, id2k_dict)

        if cluster.num_entities >= max_k * 2:
            count += 1

    return count


def get_assert_variable():
    is_assertion = rutils.str2bool(os.environ[ASSERTION_VARIABLE])
    # raise Exception("{} - {}".format(is_assertion, os.environ[ASSERTION_VARIABLE]))
    return is_assertion

def assert_invalid_clusters(clusters, id2k_dict):
    if get_assert_variable():
        count = calculate_num_invalid_clusters(clusters, id2k_dict)
        assert count == 0, "There are {} invalid clusters".format(count)


def assert_too_big_clusters(clusters, id2k_dict):
    if get_assert_variable():
        count = calculate_num_big_clusters(clusters, id2k_dict)
        assert count == 0, "There are {} big clusters".format(count)

def assert_invalid_and_big_clusters(clusters, id2k_dict):
    if get_assert_variable():
        num_invalid_clusters = calculate_num_invalid_clusters(clusters, id2k_dict)
        num_big_clusters = calculate_num_big_clusters(clusters, id2k_dict)

        assert num_invalid_clusters == 0 and num_big_clusters == 0, "There are {} invalid and {} big clusters".format(num_invalid_clusters, num_big_clusters)

def print_invalid_and_big_clusters(clusters, id2k_dict):
    if get_assert_variable():
        num_invalid_clusters = calculate_num_invalid_clusters(clusters, id2k_dict)
        num_big_clusters = calculate_num_big_clusters(clusters, id2k_dict)

        logger.info("There are {} invalid and {} big clusters".format(num_invalid_clusters, num_big_clusters))