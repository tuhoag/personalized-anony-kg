from tqdm import tqdm
import logging
import os

from .static_graph import StaticGraph

logger = logging.getLogger(__name__)

def modify_name(old_name):
    new_name = old_name.replace(",", "")

    return new_name

def load_file(graph, path):
    relation_str_set = set()
    ignored_relations_set = set()
    num_attr_rels = 0
    num_entity_rels = 0

    with open(path) as f:
        for line in f:
            splits = line.strip().split("\t")
            relation_name = splits[1][1:-1]
            node1_name = modify_name(splits[0][1:-1])
            node2_name =modify_name(splits[2][1:-1])

            # logger.debug("{}, {}, {}".format(node1_name, relation_name, node2_name))
            if is_attr_relation(relation_name):
                num_attr_rels += 1
                graph.add_attribute_edge(node1_name, relation_name, node2_name)
            elif is_entity_relation(relation_name):
                num_entity_rels += 1
                graph.add_relationship_edge(node1_name, relation_name, node2_name)
            else:
                ignored_relations_set.add(relation_name)

            relation_str_set.add(relation_name)
    logger.debug(relation_str_set)
    logger.debug("ignored relations: {}".format(ignored_relations_set))
    logger.debug("attrs: {} - entity rels: {}".format(num_attr_rels, num_entity_rels))
    # logger.debug("graph: {}".format(graph))

    # raise Exception()


def is_attr_relation(relation_name):
    attr_relations_set = {"hasWonPrize", "owns", "created", "happenedIn", "participatedIn", "directed", "diedIn", "wasBornIn", "edited", "isPoliticianOf", "graduatedFrom", "isInterestedIn", "playsFor", "wroteMusicFor", "actedIn", "isCitizenOf", "worksAt", "isLeaderOf", "isAffiliatedTo", "isKnownFor", "livesIn"}

    return relation_name in attr_relations_set

def is_entity_relation(relation_name):
    entity_relations_set = {"hasAcademicAdvisor", "hasChild", "isMarriedTo", "influences"}

    return relation_name in entity_relations_set





class YagoGraph(StaticGraph):
    @staticmethod
    def from_raw_file(data_dir, args):
        graph = YagoGraph()

        load_file(graph, os.path.join(data_dir, "yago15k_train.txt"))
        logger.debug("after train: {}".format(graph))

        load_file(graph, os.path.join(data_dir, "yago15k_test.txt"))
        logger.debug("after train + test: {}".format(graph))

        load_file(graph, os.path.join(data_dir, "yago15k_train.txt"))
        logger.debug("after train + test + valid: {}".format(graph))

        # raise Exception()
        # user_department_path = os.path.join(
        #     data_dir, 'email-Eu-core-department-labels.txt')
        # load_users_departments(graph, user_department_path)

        # user_relationship_path = os.path.join(data_dir, 'email-Eu-core.txt')
        # load_users_relationship(graph, user_relationship_path)

        return graph
