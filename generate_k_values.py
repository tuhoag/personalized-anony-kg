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
import anonygraph.k_generators as generators

logger = logging.getLogger(__file__)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_k_generator_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)


def generate_k_values(graph, generator_name, args):
    generator_fn = generators.get_generator(generator_name, args)
    entity_id2k_dict = generator_fn(graph)
    return entity_id2k_dict

def write_k_values(path, entity_id2k_dict):
    if not os.path.exists(os.path.dirname(path)):
        logger.info("creating folder: {}".format(os.path.dirname(path)))
        os.makedirs(os.path.dirname(path))

    with open(path, "w+") as f:
        for entity_id, k in entity_id2k_dict.items():
            f.write("{},{}\n".format(entity_id, k))

    logger.info("write generated k values to {}".format(path))

def main(args):
    logger.info(args)
    data_name = args["data"]
    sample = args["sample"]
    generator_name = args["gen"]

    graph = dutils.load_raw_graph(data_name, sample)

    entity_id2k_dict = generate_k_values(graph, generator_name, args)

    path = putils.get_k_values_path(data_name, sample, generator_name, args)
    write_k_values(path, entity_id2k_dict)

    logger.info("unique k values: {}".format(sorted(list(set(entity_id2k_dict.values())))))

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
