import logging

import anonygraph.info_loss.info as info

logger = logging.getLogger(__name__)

class SameAttributesGenerator:
    def __call__(self, graph, clusters):
        count = 0
        for cluster in clusters:
            union_attrs_list, entities_info = info.get_generalized_attribute_info(graph, cluster.to_list())
            logger.debug('cluster: {} - union attrs: {} - users attrs: {}'.format(
            cluster, union_attrs_list, entities_info))

            for entity_id in cluster:
                # logger.debug(union_attrs_list)
                # raise Exception()
                for relation_id, value_ids in union_attrs_list.items():
                    for value_id in value_ids:
                        graph.add_attribute_edge_from_id(entity_id, relation_id, value_id)
                        count += 1

        logger.info("added {} attributes edges".format(count))
