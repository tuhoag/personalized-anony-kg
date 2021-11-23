import networkx as nx
import logging

import anonygraph.info_loss as ifn
from anonygraph.constants import *
import anonygraph.info_loss.info as info

from anonygraph.constants import *

logger = logging.getLogger(__name__)

def calculate_num_fake_entities(anonymized_graph, graph, args):
    anonymized_entity_ids = set(anonymized_graph.entity_ids)
    raw_entity_ids = set(graph.entity_ids)

    return len(anonymized_entity_ids.difference(raw_entity_ids))


def calculate_num_real_entities(anonymized_graph, graph, args):
    anonymized_entity_ids = set(anonymized_graph.entity_ids)
    raw_entity_ids = set(graph.entity_ids)

    return len(anonymized_entity_ids.intersection(raw_entity_ids))


def calculate_num_real_edges(anonymized_graph, graph, args):
    count = 0
    for node1_id, relation_id, node2_id in anonymized_graph.get_edges_iter():
        if graph.is_edge_existed(node1_id, relation_id, node2_id):
            count += 1

    return count


def calculate_num_fake_edges(anonymized_graph, graph, args):
    count = 0
    for node1_id, relation_id, node2_id in anonymized_graph.get_edges_iter():
        if not graph.is_edge_existed(node1_id, relation_id, node2_id):
            count += 1

    return count


def calculate_num_raw_entities(anonymized_graph, graph, args):
    return graph.num_entities


def calculate_num_anonymized_entities(anonymized_graph, graph, args):
    return anonymized_graph.num_entities


def calculate_num_raw_edges(anonymized_graph, graph, args):
    return graph.num_edges


def calculate_num_anonymized_edges(anonymized_graph, graph, args):
    return anonymized_graph.num_edges

def calculate_radm(anonymized_graph, graph, args):
    anonymized_entity_ids = set(anonymized_graph.entity_ids)
    raw_entity_ids = set(graph.entity_ids)
    entity_ids = anonymized_entity_ids
    num_entities = len(entity_ids)

    info_loss = 0
    raw_entity_ids = set(graph.entity_ids)

    for entity_id in entity_ids:
        if entity_id not in raw_entity_ids:
            # fake entity
            entity_score = 1
        elif entity_id not in anonymized_entity_ids:
            # removed user
            entity_score = 1
        else:
            out_info_loss = calculate_entity_degree_info_loss(
                anonymized_graph, graph, entity_id, 'out', num_entities
            )
            in_info_loss = calculate_entity_degree_info_loss(
                anonymized_graph, graph, entity_id, 'in', num_entities
            )
            degree_info_loss = (out_info_loss + in_info_loss) / 2
            attr_info_loss = calculate_entity_attribute_info_loss(
                anonymized_graph, graph, entity_id
            )
            entity_score = (attr_info_loss + degree_info_loss) / 2

        info_loss += entity_score

    if len(entity_ids) > 0:
        info_loss = info_loss / len(entity_ids)
    else:
        info_loss = 0

    return info_loss

def calculate_adm(anonymized_graph, graph, args):
    anonymized_entity_ids = set(anonymized_graph.entity_ids)
    raw_entity_ids = set(graph.entity_ids)
    entity_ids = anonymized_entity_ids.union(raw_entity_ids)
    num_entities = len(entity_ids)

    info_loss = 0
    raw_entity_ids = set(graph.entity_ids)

    for entity_id in entity_ids:
        if entity_id not in raw_entity_ids:
            # fake entity
            entity_score = 1
        elif entity_id not in anonymized_entity_ids:
            # removed user
            entity_score = 1
        else:
            out_info_loss = calculate_entity_degree_info_loss(
                anonymized_graph, graph, entity_id, 'out', num_entities
            )
            in_info_loss = calculate_entity_degree_info_loss(
                anonymized_graph, graph, entity_id, 'in', num_entities
            )
            degree_info_loss = (out_info_loss + in_info_loss) / 2
            attr_info_loss = calculate_entity_attribute_info_loss(
                anonymized_graph, graph, entity_id
            )
            entity_score = (attr_info_loss + degree_info_loss) / 2

        info_loss += entity_score

    if len(entity_ids) > 0:
        info_loss = info_loss / len(entity_ids)
    else:
        info_loss = 0

    return info_loss


def calculate_out_in_dm(anonymized_graph, graph, args):
    anonymized_entity_ids = set(anonymized_graph.entity_ids)
    raw_entity_ids = set(graph.entity_ids)
    entity_ids = anonymized_entity_ids.union(raw_entity_ids)
    num_entities = len(entity_ids)

    info_loss = 0

    for entity_id in entity_ids:
        if entity_id not in raw_entity_ids:
            # fake entity
            entity_score = 1
        elif entity_id not in anonymized_entity_ids:
            # removed user
            entity_score = 1
        else:
            out_info_loss = calculate_entity_degree_info_loss(
                anonymized_graph, graph, entity_id, 'out', num_entities
            )
            in_info_loss = calculate_entity_degree_info_loss(
                anonymized_graph, graph, entity_id, 'in', num_entities
            )
            entity_score = (out_info_loss + in_info_loss) / 2

        info_loss += entity_score

    if len(entity_ids) > 0:
        info_loss = info_loss / len(entity_ids)
    else:
        info_loss = 0

    return info_loss



def calculate_entity_degree_info_loss(
    anonymized_graph, graph, entity_id, degree_type, num_entities
):
    anonymized_degree_info = info.get_degree_info(
        anonymized_graph, entity_id, degree_type
    )
    raw_degree_info = info.get_degree_info(graph, entity_id, degree_type)

    relationship_relation_ids = graph.relationship_relation_ids
    if len(relationship_relation_ids) == 0:
        return 0

    entity_info_loss = 0

    for relation_id in relationship_relation_ids:
        anonymized_degree = anonymized_degree_info.get(relation_id, 0)
        raw_degree = raw_degree_info.get(relation_id, 0)

        entity_info_loss += abs(anonymized_degree - raw_degree) / num_entities

    entity_info_loss /= len(relationship_relation_ids)

    return entity_info_loss


def calculate_entity_attribute_info_loss(anonymized_graph, graph, entity_id):
    anonymized_attr_info = info.get_attribute_info(anonymized_graph, entity_id)
    raw_attr_info = info.get_attribute_info(graph, entity_id)

    attribute_relation_ids = graph.attribute_relation_ids

    if len(attribute_relation_ids) == 0:
        return 0

    entity_info_loss = 0
    logger.debug("attribute_relation_ids: {}".format(attribute_relation_ids))
    # raise Exception()
    for relation_id in attribute_relation_ids:
        num_domain_value_ids = len(graph.get_domain_value_ids(relation_id))
        anonymized_value_ids = anonymized_attr_info.get(relation_id, set())
        raw_value_ids = raw_attr_info.get(relation_id, set())

        entity_info_loss += abs(len(anonymized_value_ids) - len(raw_value_ids)
                            ) / abs(num_domain_value_ids - len(raw_value_ids) + 1)

        # if len(anonymized_value_ids) > 0 or len(raw_value_ids) > 0:
        #     logger.debug("num_domain_value_ids: {}".format(num_domain_value_ids))
        #     logger.debug("anonymized_value_ids (len: {}): {}".format(len(anonymized_value_ids), anonymized_value_ids))
        #     logger.debug("raw_value_ids (len: {}): {}".format(len(raw_value_ids), raw_value_ids))


            # raise Exception()
    entity_info_loss /= len(attribute_relation_ids)

    return entity_info_loss


def calculate_am(anonymized_graph, graph, args):
    anonymized_entity_ids = set(anonymized_graph.entity_ids)
    raw_entity_ids = set(graph.entity_ids)
    entity_ids = anonymized_entity_ids.union(raw_entity_ids)

    info_loss = 0

    for entity_id in entity_ids:
        if entity_id not in raw_entity_ids:
            # fake entity
            entity_score = 1
        elif entity_id not in anonymized_entity_ids:
            # removed user
            entity_score = 1
        else:
            entity_score = calculate_entity_attribute_info_loss(
            anonymized_graph, graph, entity_id
        )

        info_loss += entity_score

    if len(entity_ids) > 0:
        info_loss = info_loss / len(entity_ids)
    else:
        info_loss = 0

    return info_loss

class calculate_anonymity:
    def __init__(self, info_type):
        self.info_type = info_type

    def _get_info_key(self, subgraph, entity_id):
        entity_info_key = None

        if self.info_type == "attr":
            entity_info_key = info.get_attribute_info_key(subgraph, entity_id)
        elif self.info_type == "out":
            entity_info_key = info.get_degree_info_key(subgraph, entity_id, "out")
        elif self.info_type == "in":
            entity_info_key = info.get_degree_info_key(subgraph, entity_id, "in")
        elif self.info_type == "out_in":
            entity_info_key = info.get_out_in_degree_info_key(subgraph, entity_id)
        elif self.info_type == "attr_out_in":
            entity_info_key = info.get_attribute_and_degree_info_key(subgraph, entity_id)
        else:
            raise Exception("Unsupported info type: {}".format(self.info_type))

        return entity_info_key

    def __call__(self, anonymized_graph, graph, args):
        anonymized_entity_ids = anonymized_graph.entity_ids

        all_entity_info_dict = {}
        for entity_id in anonymized_entity_ids:
            entity_info_key = self._get_info_key(anonymized_graph, entity_id)
            # raise Exception(entity_info)
            same_info_entity_list = all_entity_info_dict.get(entity_info_key)

            if same_info_entity_list is None:
                same_info_entity_list = []
                all_entity_info_dict[entity_info_key] = same_info_entity_list

            same_info_entity_list.append(entity_id)
            logger.debug("entity_info_dict: {}".format(all_entity_info_dict))

        if len(anonymized_entity_ids) == 0:
            anonymity_level = 0
        else:
            anonymity_level = min([len(entity_ids) for entity_ids in all_entity_info_dict.values()])
        logger.debug("entity_info_dict: {}".format(all_entity_info_dict))
        # raise Exception(all_entity_info_dict)
        return anonymity_level



def calculate_num_removed_entities(anonymized_graph, graph, args):
    anonymized_entity_ids = anonymized_graph.entity_ids
    raw_entity_ids = graph.entity_ids

    count = 0
    for entity_id in raw_entity_ids:
        if entity_id not in anonymized_entity_ids:
            count += 1

    return count

def calculate_num_removed_edges(anonymized_graph, graph, args):
    count = 0

    for node1_id, relation_id, node2_id in graph.get_edges_iter():
        if not anonymized_graph.is_edge_existed(node1_id, relation_id, node2_id):
            count += 1

    return count

subgraph_metric_dict = {
    ADM_METRIC: calculate_adm,
    DM_METRIC: calculate_out_in_dm,
    AM_METRIC: calculate_am,
    FAKE_EDGES_METRIC: calculate_num_fake_edges,
    REMAINING_ADM_METRIC: calculate_radm,
    RAW_ENTITIES_METRIC: calculate_num_raw_entities,
    ANONYMIZED_ENTITIES_METRIC: calculate_num_anonymized_entities,
    RAW_EDGES_METRIC: calculate_num_raw_edges,
    ANONYMIZED_EDGES_METRIC: calculate_num_anonymized_edges,
    ANONYMIZED_ANONYMITY_METRIC: calculate_anonymity("attr_out_in"),
    ANONYMIZED_ATTRIBUTE_ANONYMITY_METRIC: calculate_anonymity("attr"),
    ANONYMIZED_DEGREE_ANONYMITY_METRIC: calculate_anonymity("out_in"),
    REMOVED_ENTITIES_METRIC: calculate_num_removed_entities,
    REMOVED_EDGES_METRIC: calculate_num_removed_edges,
    FAKE_ENTITIES_METRIC: calculate_num_fake_entities,
}


def get_all_metric_names():
    return list(subgraph_metric_dict.keys())


def calculate_quality_metrics(metric_names, anonymized_graph, graph, args):
    quality = {}
    for metric_name in metric_names:
        fn = subgraph_metric_dict[metric_name]
        quality_value = fn(anonymized_graph, graph, args)
        quality[metric_name] = quality_value

    return quality
