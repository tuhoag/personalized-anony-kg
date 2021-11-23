import logging
from tensorflow.keras.utils import Sequence
import math
import numpy as np

import anonygraph.utils.data as dutils

logger = logging.getLogger(__name__)


class PairDistanceSequence(Sequence):
    def __init__(self, data_name, sample, info_loss_name, batch_size, args):
        self.data_name = data_name
        self.sample = sample
        self.info_loss_name = info_loss_name
        self.batch_size = batch_size
        self.dist_matrix = dutils.load_dist_matrix(
            data_name, sample, info_loss_name, args
        )

    @property
    def num_entities(self):
        """
        Get the number of entities.
        """
        return self.dist_matrix.shape[0]

    @property
    def num_pairs(self):
        """
        Get the number of pairs.
        """
        return self.num_entities ** 2

    def __len__(self):
        """
        Get the number of batches.
        """
        # num_entities = self.num_entities
        # logger.debug(num_entities)
        # num_pairs = num_entities**2
        # logger.debug(num_pairs)
        num_batches = math.ceil(self.num_pairs / self.batch_size)
        logger.debug(num_batches)

        return num_batches

    def __getitem__(self, index):
        """
        Get the batch of the given index.
        """
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_pairs)
        # logger.debug("start idx: {} - end idx: {}".format(start_idx, end_idx))

        idxs = range(start_idx, end_idx)
        # logger.debug("idxs: {}".format(list(idxs)))

        entity1_idxs = list(map(lambda idx: math.floor(idx/self.num_entities), idxs))
        entity2_idxs = list(map(lambda idx: idx%self.num_entities, idxs))
        # logger.debug("entity1 ids: {}".format(entity1_idxs))
        # logger.debug("entity2 ids: {}".format(entity2_idxs))

        batch_x = np.column_stack((entity1_idxs, entity2_idxs))
        batch_y = np.fromiter(map(lambda pair: self.dist_matrix[pair[0], pair[1]] ** 2, batch_x), dtype=np.float)

        # logger.debug("start idx: {} - end idx: {}".format(start_idx, end_idx))
        # logger.debug("idxs: {}".format(idxs))
        # logger.debug("entity1 ids: {}".format(entity1_idxs))
        # logger.debug("entity2 ids: {}".format(entity2_idxs))
        # logger.debug("batch x - shape {}: {}".format(batch_x.shape, batch_x))
        # logger.debug("batch y - shape {}: {}".format(batch_y.shape, batch_y))

        return batch_x, batch_y

    def __str__(self):
        """
        Get description of the sequence.
        """
        return """Data: {} (Sample: {})
            Number of entities: {}
            Number of pairs: {}
            Batch size: {}
            Number of batches: {}""".format(
            self.data_name, self.sample, self.num_entities,
            self.num_entities**2, self.batch_size, len(self)
        )