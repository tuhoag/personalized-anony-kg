class BaseHandler:
    def __init__(self, name):
        self.__name = name

    def __call__(self, dist_matrix, entity_id2idx_dict, entity_idx2id_dict, entity_id2k_dict):
        """Remove outliers and return remaining entities' id.
        """
        raise NotImplementedError()

