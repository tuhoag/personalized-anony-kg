import numpy as np
from tqdm import tqdm
from glob import glob
import argparse
import logging
import os
import itertools
from joblib import Parallel, delayed
import tensorflow as tf

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


def _float_feature(value):
    """"Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_pair_dist_sample(entity1_id, entity2_id, distance):
    feature = {
        "node1_id": _int64_feature(entity1_id),
        "node2_id": _int64_feature(entity2_id),
        "distance": _float_feature(distance),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def generate_pair_distances_for_graph(data_name, sample, info_loss_name, args):
    # load subgraph
    graph = dutils.load_raw_graph(data_name, sample)
    logger.debug(graph)

    # load info loss
    info_loss_fn = ifn.get_info_loss_function(info_loss_name, graph, args)

    # get relative ids
    entity_idx2id = dutils.load_entity_idx2id_dict(data_name, sample)
    logger.debug(entity_idx2id)

    dist_pairs_path = putils.get_distance_pairs_path(data_name, sample, info_loss_name, "tfrecord", args)

    logger.debug(dist_pairs_path)
    if not os.path.exists(os.path.dirname(dist_pairs_path)):
        logger.info("creating folder: {}".format(os.path.dirname(dist_pairs_path)))
        os.makedirs(os.path.dirname(dist_pairs_path))

    # generate distance matrix
    # init dist matrix
    num_entities = len(entity_idx2id)
    entity_idxes = list(entity_idx2id.keys())

    with tf.io.TFRecordWriter(dist_pairs_path) as writer:
        with tqdm(total=num_entities*num_entities) as pbar:
            for entity1_idx, entity2_idx in itertools.combinations(entity_idxes, r=2):
                entity1_id = entity_idx2id[entity1_idx]
                entity2_id = entity_idx2id[entity2_idx]

                info_loss_val = info_loss_fn.call({entity1_id, entity2_id})

                if entity1_id != entity2_id:
                    writer.write(serialize_pair_dist_sample(entity1_idx, entity2_idx, info_loss_val))
                    pbar.update(1)

                writer.write(serialize_pair_dist_sample(entity2_idx, entity1_idx, info_loss_val))

                pbar.update(1)


def main(args):
    logger.info(args)

    generate_pair_distances_for_graph(args["data"], args["sample"], args["info_loss"], args)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
