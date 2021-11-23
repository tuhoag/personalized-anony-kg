import logging

logger = logging.getLogger(__name__)

from .base_info_loss import BaseInfoLossMetric
from .info import get_generalized_attribute_info

class AttributeInfoLoss(BaseInfoLossMetric):
    def call(self, entity_ids):
        score = 0.0

        if len(entity_ids) == 0:
            return score

        union_info, entities_info = get_generalized_attribute_info(self.graph, entity_ids)
        num_attributes = self.graph.num_attribute_relations

        logger.debug("union: {} - entities ({}): {} ".format(union_info, entity_ids, entities_info))
        if len(union_info) != 0:
            num_real_entities = 0
            for entity_id, entity_info in entities_info.items():
                if self.graph.is_entity_id(entity_id):
                    num_real_entities += 1
                    entity_score = 0.0

                    for relation_id, union_value_ids in union_info.items():
                        max_num_value_ids = self.graph.get_num_domain_value_ids_from_relation_id(
                            relation_id)
                        num_value_ids = len(union_value_ids)
                        num_entity_value_ids = len(entity_info.get(relation_id, set()))
                        if max_num_value_ids - num_entity_value_ids + 1 == 0:
                            raise Exception("relation: {} union: {} domain: {}".format(relation_id, union_value_ids, self.graph.get_domain_value_ids(relation_id)))
                            raise Exception("max: {} - num: {}".format(max_num_value_ids, num_entity_value_ids))
                        current_score = (num_value_ids - num_entity_value_ids) / (max_num_value_ids - num_entity_value_ids + 1)

                        entity_score += current_score

                        logger.debug("relation: {} union: {} - info: {} - max: {} - score: {}".format(relation_id, num_value_ids, num_entity_value_ids, max_num_value_ids, current_score))

                    score += entity_score / num_attributes
                    logger.debug("cluster: {} - user: {} - score: {}".format(entity_ids, entity_info, entity_score / num_attributes))
                # print(score)
            score = score / num_real_entities

            # print("clusters: {} - md - score: {}".format(user_ids_set, score))
        return score