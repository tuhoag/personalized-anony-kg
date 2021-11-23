import numpy as np
from tqdm import tqdm
from glob import glob
import argparse
import logging
import os
import itertools
from joblib import Parallel, delayed

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.info_loss as ifn

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_info_loss_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

def generate_pair_distances_for_graph(data_name, sample, info_loss_name, args):
    # load subgraph
    graph = dutils.load_raw_graph(data_name, sample)
    logger.debug(graph)

    # load info loss
    info_loss_fn = ifn.get_info_loss_function(info_loss_name, graph, args)

    # get relative ids
    entity_idx2id = dutils.load_entity_idx2id_dict(data_name, sample)
    logger.debug(entity_idx2id)

    dist_matrix_path = putils.get_distance_matrix_path(data_name, sample, info_loss_name, args)
    logger.debug(dist_matrix_path)
    if not os.path.exists(os.path.dirname(dist_matrix_path)):
        logger.info("creating folder: {}".format(os.path.dirname(dist_matrix_path)))
        os.makedirs(os.path.dirname(dist_matrix_path))

    # generate distance matrix
    # init dist matrix
    num_entities = len(entity_idx2id)
    entity_idxes = list(entity_idx2id.keys())

    dist_matrix = np.zeros(shape=(num_entities, num_entities))

    count = 0
    with tqdm(total=num_entities*num_entities) as pbar:
        for entity1_idx, entity2_idx in itertools.combinations(entity_idxes, r=2):
            entity1_id = entity_idx2id[entity1_idx]
            entity2_id = entity_idx2id[entity2_idx]

            info_loss_val = info_loss_fn.call({entity1_id, entity2_id})
            dist_matrix[entity1_idx, entity2_idx] = info_loss_val
            dist_matrix[entity2_idx, entity1_idx] = info_loss_val

            count += 2
            pbar.update(2)

    logger.info("calculated {} dists".format(count))
    np.save(dist_matrix_path, dist_matrix)


def main(args):
    logger.info(args)

    generate_pair_distances_for_graph(args["data"], args["sample"], args["info_loss"], args)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
