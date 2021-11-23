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
    rutils.add_learner_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)


def get_model(num_nodes, num_dimensions):
    """
    Initialize a model with the given number of dimensions.
    """
    model = SquaredEuclideanDistanceModel(num_nodes, num_dimensions)
    model.build((None, 2))
    return model


def test_dataset(dataset):
    for idx, temp in enumerate(dataset):
        print("{} - {}".format(idx, temp))
    # print(dataset[8])
    # print(dataset[1])


def get_dataset(data_name, sample, info_loss_name, batch_size, args):
    """
    Get the dataset of pairwise distance.
    """
    dataset = PairDistanceSequence(
        data_name, sample, info_loss_name, batch_size, args
    )
    return dataset


def train(model, dataset, num_epochs, num_workers, gpu, is_continue, args):
    """
    Train the given model with the given dataset.
    """
    log_dir_path = ""
    checkpoint_dir_path = putils.get_model_checkpoint_path(
        dataset.data_name, dataset.sample, dataset.info_loss_name,
        model.num_dimensions, args
    )

    logger.debug("checkpoint path: {}".format(checkpoint_dir_path))
    # raise Exception()

    if is_continue:
        logger.info("loaded checkpoint from: {}".format(checkpoint_dir_path))
        model.load_weights(checkpoint_dir_path)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[MeanSquaredRootAbsoluteError()],
    )
    model.fit(
        x=dataset,
        epochs=num_epochs,
        workers=num_workers,
        callbacks=[
            # callbacks.EarlyStopping(),
            # callbacks.TensorBoard(log_dir=log_dir_path),
            callbacks.ModelCheckpoint(
                checkpoint_dir_path,
                verbose=0,
                save_weights_only=False
            ),
        ]
    )
    # model.evaluate(
    #     x=dataset,
    #     workers=num_workers,
    #     callbacks=[
    #         # callbacks.EarlyStopping(),
    #         # callbacks.TensorBoard(log_dir=log_dir_path),
    #         callbacks.ModelCheckpoint(
    #             checkpoint_dir_path,
    #             verbose=0,
    #             save_weights_only=True
    #         ),
    #     ]
    # )


def main(args):
    logger.info(args)
    data_name = args["data"]
    sample = args["sample"]
    num_dimensions = args["d"]
    info_loss_name = args["info_loss"]
    batch_size = args["batch"]
    num_workers = args["workers"]
    num_epochs = args["epochs"]
    is_continue = args["continue"]

    gpu = args["gpu"]
    num_nodes = dutils.get_number_of_entities(data_name, sample)
    logger.debug("num_nodes: {}".format(num_nodes))

    # create model
    model = get_model(num_nodes, num_dimensions)
    logger.debug("loaded model: {}".format(model.summary()))

    # create dataset
    dataset = get_dataset(data_name, sample, info_loss_name, batch_size, args)
    logger.debug("dataset: {}".format(dataset))

    # test_dataset(dataset)

    # train
    train(model, dataset, num_epochs, num_workers, gpu, is_continue, args)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
