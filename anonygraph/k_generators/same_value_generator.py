class SameValueGenerator:
    def __init__(self, value):
        self.value = value

    def __call__(self, graph):
        entity_ids = graph.entity_ids

        entity_id2k_dict = {}
        for entity_id in entity_ids:
            entity_id2k_dict[entity_id] = self.value

        return entity_id2k_dict