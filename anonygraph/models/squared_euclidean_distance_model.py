
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)
class SquaredEuclideanDistanceModel(keras.Model):
    def __init__(self, num_nodes, num_dimensions):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_dimensions = num_dimensions

        self.embedding = layers.Embedding(num_nodes, num_dimensions)

    def call(self, inputs):
        logger.debug("input shape: {}".format(inputs.shape))
        logger.debug("inputs: {}".format(inputs))
        # print(inputs)
        # raise Exception()
        node1_ids = inputs[:,0]
        node2_ids = inputs[:,1]

        logger.debug("node 1 shape: {}".format(node1_ids.shape))

        node1_points = self.embedding(node1_ids)
        node2_points = self.embedding(node2_ids)

        logger.debug("node 1 points shape: {}".format(node1_points.shape))

        distance = tf.math.reduce_sum(tf.math.square(node1_points - node2_points), axis=1)

        # assert(distance.shape.as_list() == [inputs.shape[0],]), "Expected output shape: {}. Given {}".format((inputs.shape[0],), distance.shape)

        return distance