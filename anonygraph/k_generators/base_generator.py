from abc import abstractmethod

class BaseGenerator:
    def __init__(self):
        pass

    @abstractmethod
    def generate_k_values(self, entity_ids, graph):
        raise NotImplementedError("Should implement generate_k_values function")


    def __call__(self, graph):
        entity_ids = list(graph.entity_ids)

        random_k_vals = self.generate_k_values(entity_ids, graph)

        entity_id2k_dict = {}

        for entity_idx, k in enumerate(random_k_vals):
            entity_id2k_dict[entity_ids[entity_idx]] = k

        return entity_id2k_dict