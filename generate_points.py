import tensorflow as tf
import numpy as np
from tqdm import tqdm
from glob import glob
import argparse
import logging
import os
import itertools
from joblib import Parallel, delayed
from tensorflow.keras import callbacks

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
from anonygraph.models import SquaredEuclideanDistanceModel
from anonygraph.data import PairDistanceSequence
from anonygraph.evaluation.model_metrics import MeanSquaredRootAbsoluteError

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_info_loss_argument(parser)
    rutils.add_points_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)


def get_model(num_nodes, num_dimensions):
    """
    Initialize a model with the given number of dimensions.
    """
    model = SquaredEuclideanDistanceModel(num_nodes, num_dimensions)
    model.build((None, 2))
    return model


def generate_points(data_name, sample, info_loss_name, model, args):
    # load model
    checkpoint_dir_path = putils.get_model_checkpoint_path(
        data_name, sample, info_loss_name,
        model.num_dimensions, args
    )

    model.load_weights(checkpoint_dir_path)

    points = model.embedding.weights[0].numpy()
    return points


def main(args):
    logger.info(args)
    data_name = args["data"]
    sample = args["sample"]
    num_dimensions = args["d"]
    info_loss_name = args["info_loss"]

    num_nodes = dutils.get_number_of_entities(data_name, sample)
    logger.debug("num_nodes: {}".format(num_nodes))

    # create model
    model = get_model(num_nodes, num_dimensions)
    logger.debug("loaded model: {}".format(model.summary()))

    # extract points
    points = generate_points(data_name, sample, info_loss_name, model, args)

    points_path = putils.get_points_path(data_name, sample, info_loss_name, num_dimensions, args)

    if not os.path.exists(os.path.dirname(points_path)):
        logger.info("created folder: {}".format(os.path.dirname(points_path)))
        os.makedirs(os.path.dirname(points_path))

    np.save(points_path, points)

    logger.info("saved points (shape {}) to: {}".format(points.shape, points_path))

if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
