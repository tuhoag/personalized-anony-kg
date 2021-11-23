import logging
import numpy as np
from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)

class StaticGenerator(BaseGenerator):
    def __init__(self, data_name):
        if data_name != "dummy":
            raise Exception()

        self.data_name = data_name

    def generate_k_values(self, entity_ids, graph):
        return np.array([2, 2, 2, 1, 4])