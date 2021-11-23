from anonygraph.constants import *
from .base_handler import BaseHandler

class NoRemovalHandler(BaseHandler):
    def __init__(self):
        super().__init__(NO_REMOVAL_HANDLER)

    def __call__(self, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
        return list(entity_id2idx_dict.keys())