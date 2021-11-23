import os
import logging

import networkx as nx

logger = logging.getLogger(__name__)

ATTRIBUTE_RELATION_TYPE = "attr"
RELATIONSHIP_RELATION_TYPE = "rel"


class StaticGraph(object):
    def __init__(self):
        self.__graph = nx.MultiDiGraph()

        # take from dynamic graph
        self.__node2id = {}
        self.__relation2id = {}
        self.__relationship_relation_ids = set()
        self.__attribute_relation_ids = set()
        self.__attribute_domains = {}

        # nodes exist in this subgraph
        self.__entity_ids = set()
        self.__value_ids = set()

        self.__num_relationship_edges = 0
        self.__num_attribute_edges = 0

    @property
    def relationship_relation_ids(self):
        return self.__relationship_relation_ids

    @property
    def attribute_relation_ids(self):
        return self.__attribute_relation_ids


    @property
    def num_nodes(self):
        return len(self.__node2id)

    @property
    def num_entities(self):
        return len(self.__entity_ids)

    @property
    def num_values(self):
        return len(self.__value_ids)

    @property
    def num_relations(self):
        return len(self.__relation2id)

    @property
    def num_attribute_relations(self):
        return len(self.__attribute_relation_ids)

    @property
    def num_relationship_relations(self):
        return len(self.__relationship_relation_ids)

    @property
    def num_edges(self):
        return self.__graph.number_of_edges()

    @property
    def num_attribute_edges(self):
        return self.__num_attribute_edges

    @property
    def num_relationship_edges(self):
        return self.__num_relationship_edges

    @property
    def entity_ids(self):
        return self.__entity_ids

    def get_domain_value_ids(self, relation_id):
        return self.__attribute_domains[relation_id]

    def __add_edge_from_id(self, node1_id, relation_id, node2_id):
        if self.is_attribute_relation_id(relation_id):
            self.__num_attribute_edges += 1
        else:
            self.__num_relationship_edges += 1

        self.__graph.add_edge(node1_id, node2_id, key=relation_id)

    def remove_edge_from_id(self, node1_id, relation_id, node2_id):
        if self.is_attribute_relation_id(relation_id):
            self.__num_attribute_edges -= 1
        else:
            self.__num_relationship_edges -= 1

        self.__graph.remove_edge(node1_id, node2_id, key=relation_id)

    def add_relationship_edge_from_id(self, entity1_id, relation_id, entity2_id):
        self.__entity_ids.add(entity1_id)
        self.__entity_ids.add(entity2_id)
        self.__relationship_relation_ids.add(relation_id)

        self.__add_edge_from_id(entity1_id, relation_id, entity2_id)

    def add_attribute_domain_from_id(self, relation_id, value_id):
        domain = self.__attribute_domains.get(relation_id)

        if domain is None:
            domain = set()
            self.__attribute_domains[relation_id] = domain

        domain.add(value_id)

    def add_attribute_edge_from_id(self, entity_id, relation_id, value_id):
        self.__entity_ids.add(entity_id)
        self.__value_ids.add(value_id)
        self.__attribute_relation_ids.add(relation_id)

        self.add_attribute_domain_from_id(relation_id, value_id)

        self.__add_edge_from_id(entity_id, relation_id, value_id)

    def add_attribute_edge(self, entity_name, relation_name, value_name):
        entity_id = self.get_node_id(entity_name)
        value_id = self.get_node_id(value_name)
        relation_id = self.get_relation_id(relation_name, ATTRIBUTE_RELATION_TYPE)

        self.add_attribute_edge_from_id(entity_id, relation_id, value_id)

    def add_relationship_edge(self, entity1_name, relation_name, entity2_name):
        entity1_id = self.get_node_id(entity1_name)
        entity2_id = self.get_node_id(entity2_name)
        relation_id = self.get_relation_id(relation_name, RELATIONSHIP_RELATION_TYPE)

        self.add_relationship_edge_from_id(
            entity1_id, relation_id, entity2_id)

    def is_attribute_relation_id(self, relation_id):
        return relation_id in self.__attribute_relation_ids

    def is_relationship_relation_id(self, relation_id):
        return relation_id in self.__relationship_relation_ids

    def get_node_id(self, name):
        entity_id = _get_item_id_from_name(self.__node2id, name)
        return entity_id

    def get_relation_id(self, name, relation_type):
        raw_name = "{}_{}".format(name, relation_type)
        relation_id = _get_item_id_from_name(self.__relation2id, raw_name)
        return relation_id

    def get_edges_iter(self):
        for entity1_id, entity2_id, relation_id in self.__graph.edges(keys=True):
            yield entity1_id, relation_id, entity2_id

    def export(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        self.__export_index(path)
        self.__export_domains(path)
        self.__export_edges(path)

    def __export_index(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        entityidx_path = os.path.join(path, 'entities.idx')
        valueidx_path = os.path.join(path, 'values.idx')
        attrsidx_path = os.path.join(path, 'attrs.idx')
        relsidx_path = os.path.join(path, 'rels.idx')

        write_index_file(valueidx_path, self.__value_ids, self.__node2id)
        write_index_file(entityidx_path, self.__entity_ids, self.__node2id)
        write_index_file(attrsidx_path, self.__attribute_relation_ids,
                    self.__relation2id)
        write_index_file(relsidx_path, self.__relationship_relation_ids,
                    self.__relation2id)

    def __export_domains(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        domains_path = os.path.join(path, 'domains.txt')

        with open(domains_path, 'w') as f:
            for relation_id, domain in self.__attribute_domains.items():
                domain_str = ""

                for value_id in domain:
                    domain_str += "{},".format(value_id)

                line = "{}:{}\n".format(relation_id, domain_str[:-1])
                f.write(line)


    def __export_edges(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        logger.debug("writing edges to: {}".format(path))
        rel_info_path = os.path.join(path, 'rels.edges')
        attr_info_path = os.path.join(path, 'attrs.edges')

        # save metadata
        rel_info_file = open(rel_info_path, 'w')
        attr_info_file = open(attr_info_path, 'w')

        for entity1_id, relation_id, entity2_id in self.get_edges_iter():
            line = '{},{},{}\n'.format(
                entity1_id, relation_id, entity2_id)
            # logger.debug(line)
            if self.is_attribute_relation_id(relation_id):
                # logger.debug('write to attr file')
                current_file = attr_info_file
            else:
                # logger.debug('write to rel file')
                current_file = rel_info_file

            current_file.write(line)

        rel_info_file.close()
        attr_info_file.close()

    @staticmethod
    def from_raw_graph_output(path):
        graph = StaticGraph()

        node2id, relation2id, user_ids, value_ids, attribute_relation_ids, relationship_relation_ids = read_index_data(path)
        # logger.debug(relation2id)
        graph.__node2id = node2id
        graph.__relation2id = relation2id
        graph.__entity_ids = user_ids
        graph.__value_ids = value_ids
        graph.__attribute_relation_ids = attribute_relation_ids
        graph.__relationship_relation_ids = relationship_relation_ids

        graph.__attribute_domains = read_domains_data(path)

        read_edges(path, graph)

        return graph

    def get_attribute_edges_iter_of_entity_id(self, entity_id):
        if self.is_entity_id(entity_id):
            for _, value_id, relation_id in self.__graph.out_edges(entity_id, keys=True):
                if self.is_attribute_relation_id(relation_id):
                    yield entity_id, relation_id, value_id

    def get_out_relationship_edges_iter_of_entity_id(self, entity1_id):
        if self.is_entity_id(entity1_id):
            for _, entity2_id, relation_id in self.__graph.out_edges(entity1_id, keys=True):
                if not self.is_attribute_relation_id(relation_id):
                    yield entity1_id, relation_id, entity2_id

    def get_in_relationship_edges_iter_of_entity_id(self, entity_id):
        if self.is_entity_id(entity_id):
            for entity2_id, _, relation_id in self.__graph.in_edges(entity_id, keys=True):
                yield entity2_id, relation_id, entity_id

    def get_num_domain_value_ids_from_relation_id(self, relation_id):
        domain_value_ids = self.__attribute_domains.get(relation_id)
        return len(domain_value_ids)

    def is_entity_id(self, entity_id):
        return entity_id in self.__entity_ids

    def is_edge_existed(self, node1_id, relation_id, node2_id):
        # for tnode1_id, trelation_id, tnode2_id in self.get_edges_iter():
        #     if trelation_id == relation_id:
        #         logger.debug("({}, {}, {})".format(tnode1_id, trelation_id, tnode2_id))

        is_existed = self.__graph.has_edge(node1_id, node2_id, key=relation_id)

        # logger.debug(is_existed)
        # raise Exception()
        return is_existed

    def __str__(self):
        return ("number of nodes: {:,d} (entities: {:,d} - values: {:,d})\n"
                "number of relations: {:,d} (relationships: {:,d} - attributes: {:,d})\n"
                "number of edges: {:,d}(relationships: {:,d} - attributes: {:,d})".format(
                                                        self.num_nodes,
                                                        self.num_entities,
                                                        self.num_values,
                                                        self.num_relations,
                                                        self.num_relationship_relations,
                                                        self.num_attribute_relations,
                                                        self.num_edges,
                                                        self.num_relationship_edges,
                                                        self.num_attribute_edges))

    def remove_entity_id(self, entity_id):
        self.__entity_ids.remove(entity_id)
        self.__graph.remove_node(entity_id)

    def to_edge_files(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        rel_info_path = os.path.join(path, "rels.edges")
        attr_info_path = os.path.join(path, "attrs.edges")

        with open(rel_info_path, "w") as rel_info_file, open(attr_info_path, "w") as attr_info_file:
            for entity1_id, entity2_id, relation_id in self.__graph.edges(keys=True):
                line = "{},{},{}\n".format(
                    entity1_id, relation_id, entity2_id)
                # logger.debug(line)
                if self.is_attribute_relation_id(relation_id):
                    # logger.debug("write to attr file")
                    current_file = attr_info_file
                elif self.is_relationship_relation_id(relation_id):
                    # logger.debug("write to rel file")
                    current_file = rel_info_file

                current_file.write(line)

    @staticmethod
    def from_index_data(node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains):
        graph = StaticGraph()

        graph.__node2id = node2id
        graph.__relation2id = relation2id
        graph.__relationship_relation_ids = relationship_relation_ids
        graph.__attribute_relation_ids = attribute_relation_ids
        graph.__attribute_domains = attribute_domains

        return graph

    @staticmethod
    def from_index_and_edges_data(node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains, attribute_edges, relationship_edges):
        """Load Subgraph from index of dynamic graph and its edges data.

        Arguments:
            node2id {[type]} -- [description]
            relation2id {[type]} -- [description]
            relationship_relation_ids {[type]} -- [description]
            attribute_relation_ids {[type]} -- [description]
            attribute_domains {[type]} -- [description]
            attribute_edges {[type]} -- [description]
            relationship_edges {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        graph = StaticGraph.from_index_data(
            node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains)

        for node1_id, relation_id, node2_id in attribute_edges:
            # logger.debug("add attribute edge: {}, {}, {}".format(node1_id, relation_id, node2_id))
            graph.add_attribute_edge_from_id(node1_id, relation_id, node2_id)
            # if relation_id == 11:
            #     raise Exception()
            # logger.debug(graph)
            # logger.debug("entities: {} - values: {}".format(graph.__entity_ids, graph.__value_ids))


        for node1_id, relation_id, node2_id in relationship_edges:
            # logger.debug("add relationship edge: {}, {}, {}".format(node1_id, relation_id, node2_id))
            graph.add_relationship_edge_from_id(
                node1_id, relation_id, node2_id)

            # logger.debug("entities: {} - values: {}".format(graph.__entity_ids, graph.__value_ids))
            # logger.debug(graph)

        return graph



def write_index_file(path, ids, name2id):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        for id in ids:
            name = _get_name_from_id(id, name2id)
            f.write("{},{}\n".format(name, id))

def _get_name_from_id(id, name2id):
    for item_name, item_id in name2id.items():
        if id == item_id:
            return item_name

    return None

def _get_item_id_from_name(name2id, name):
    item_id = name2id.get(name, None)

    if item_id is None:
        item_id = len(name2id)
        name2id[name] = item_id

    return item_id

def read_index_data(path):
    users_idx_path = os.path.join(path, 'entities.idx')
    values_idx_path = os.path.join(path, 'values.idx')
    attrs_idx_path = os.path.join(path, 'attrs.idx')
    rels_idx_path = os.path.join(path, 'rels.idx')

    node2id = read_index_file([users_idx_path])
    user_ids = set(node2id.values())

    value2id = read_index_file([values_idx_path])
    value_ids = set(value2id.values())

    # raise Exception("{} {}".format(user_ids, value_ids))
    node2id.update(value2id)

    relation2id = read_index_file([attrs_idx_path])
    attribute_relation_ids = set(relation2id.values())

    relationship2id = read_index_file([rels_idx_path])
    relationship_relation_ids = set(relationship2id.values())

    relation2id.update(relationship2id)

    # raise Exception("{} \n{} \n{} \n{} \n{} \n{}".format(node2id, relation2id, user_ids, value_ids, attribute_relation_ids, relationship_relation_ids))
    return node2id, relation2id, user_ids, value_ids, attribute_relation_ids, relationship_relation_ids

def read_index_file(file_paths):
    result = {}

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                # logger.debug('line: {}'.format(line))
                splits = line.rstrip().split(',')
                name = ','.join(splits[0:-1])
                idx = int(splits[-1])

                # logger.debug('name = {} - idx = {}'.format(name, idx))
                # name, idx = splits[0], int(splits[1])
                result[name] = idx

    return result

def read_edges(path, graph):
    attr_edges_path = os.path.join(path, 'attrs.edges')
    rel_edges_path = os.path.join(path, 'rels.edges')

    read_edges_file(attr_edges_path, graph, lambda graph, node1_id, relation_id,
                     node2_id: graph.add_attribute_edge_from_id(node1_id, relation_id, node2_id))
    read_edges_file(rel_edges_path, graph, lambda graph, node1_id, relation_id, node2_id: graph.add_relationship_edge_from_id(node1_id, relation_id, node2_id))

def read_edges_file(path, graph, edge_addition_fn):
    with open(path, 'r') as f:
        for line in f:
            splits = line.strip().split(',')
            entity_id, relation_id, node_id = int(splits[0]), int(
                splits[1]), int(splits[2])

            edge_addition_fn(graph, entity_id, relation_id, node_id)

def read_domains_data(path):
    domain_path = os.path.join(path, 'domains.txt')

    domains_data = {}
    with open(domain_path, 'r') as f:
        for line in f:
            relation_id, domain_str = line.strip().split(':')
            domain_value_ids = set(map(int, domain_str.split(',')))
            domains_data[int(relation_id)] = domain_value_ids

    return domains_data