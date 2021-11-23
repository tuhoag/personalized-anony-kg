import logging

logger = logging.getLogger(__name__)

class UnusedEntitiesRemovalGenerator:
    def __call__(self, graph, clusters):
        logger.debug("clusters: {}".format(clusters))
        entity_ids_in_graph = list(graph.entity_ids)
        num_entities_before_removals = graph.num_entities

        num_removed_entities = 0
        for entity_id in entity_ids_in_graph:
            if not clusters.has_entity_id(entity_id):
                graph.remove_entity_id(entity_id)
                logger.debug("remove entity: {}".format(entity_id))
                num_removed_entities += 1

        logger.info("removed {} entities".format(num_removed_entities))
        num_entities_after_removals = graph.num_entities
        # logger.debug("num entities before removal: {}".format(num_entities_before_removals))
        # logger.debug("num entities after removal: {}".format(num_entities_after_removals))
        assert num_entities_before_removals - num_entities_after_removals == num_removed_entities
