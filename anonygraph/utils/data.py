import os
from anonygraph import algorithms
import logging
import numpy as np

import anonygraph.data as data
import anonygraph.utils.path as putils


logger = logging.getLogger(__name__)


def load_graph_from_raw_data(data_name, sample, args):
    raw_dir = putils.get_raw_data_path(data_name)
    data_fn_dict = {
        # "email-temp": data.EmailTempGraph,
        "dummy": data.DummyGraph,
        # "yago": data.YagoGraph,
        # "icews14": data.ICEWS14Graph,
        # "dblp": data.DBLPGraph,
        "freebase": data.FreebaseGraph,
        # "gplus": data.GplusGraph,
        "email": data.EmailGraph,
        "yago": data.YagoGraph,
        "german": data.GermanCreditGraph,
    }

    data_fn = data_fn_dict.get(data_name)
    if data_fn is not None:
        graph = data_fn.from_raw_file(raw_dir, args)
    else:
        raise NotImplementedError("Unsupported graph: {}".format(data_name))

    return graph

def load_graph_metadata(data_name, sample):
    raw_graph_path = putils.get_raw_graph_path(data_name, sample)
    node2id, relation2id, _, _, attribute_relation_ids, relationship_relation_ids = data.static_graph.read_index_data(
        raw_graph_path)
    attribute_domains = data.static_graph.read_domains_data(
        raw_graph_path)

    return node2id, relation2id, attribute_relation_ids, relationship_relation_ids, attribute_domains

def get_edges_iter(path):
    with open(path, "r") as file:
        for line in file:
            entity1_id, relation_id, entity2_id = list(map(int, line.strip().split(",")))
            yield entity1_id, relation_id, entity2_id


def load_edges_iter_from_path(path):
    rel_info_path = os.path.join(path, "rels.edges")
    attr_info_path = os.path.join(path, "attrs.edges")

    attribute_edges_iter = get_edges_iter(attr_info_path)
    relationship_edges_iter = get_edges_iter(rel_info_path)

    return attribute_edges_iter, relationship_edges_iter

def load_anonymized_graph_from_path(data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args):
    node2id, relation2id, attribute_relation_ids, relationship_relation_ids, attribute_domains = load_graph_metadata(data_name, sample)

    path = putils.get_anony_graph_path(data_name, sample, k_generator_name, info_loss_name, handler_name, calgo_name, enforcer_name, args)

    attribute_edges_iter, relationship_edges_iter = load_edges_iter_from_path(path)

    return data.StaticGraph.from_index_and_edges_data(node2id, relation2id, relationship_relation_ids, attribute_relation_ids, attribute_domains, attribute_edges_iter, relationship_edges_iter)

def load_raw_graph(data_name, sample):
    path = putils.get_raw_graph_path(data_name, sample)

    return data.StaticGraph.from_raw_graph_output(path)

def load_entity_idx2id_dict(data_name, sample):
    idx_path = putils.get_entity_index_path(data_name, sample)
    idx2id_dict = {}

    with open(idx_path, "r") as f:
        for idx, line in enumerate(f):
            _,entity_id = line.rstrip().split(",")
            idx2id_dict[idx] = int(entity_id)

    return idx2id_dict

def load_entity_id2idx_dict(data_name, sample):
    idx_path = putils.get_entity_index_path(data_name, sample)
    id2idx_dict = {}

    with open(idx_path, "r") as f:
        for idx, line in enumerate(f):
            _,entity_id = line.rstrip().split(",")
            id2idx_dict[int(entity_id)] = idx

    return id2idx_dict

def load_dist_matrix(data_name, sample, info_loss_name, args):
    path = putils.get_distance_matrix_path(data_name, sample, info_loss_name, args)
    return np.load(path)

def load_entity_id2k_dict(data_name, sample, generator_name, args):
    id2k_dict_path = putils.get_k_values_path(data_name, sample, generator_name, args)

    id2k_dict = {}

    with open(id2k_dict_path, "r") as f:
        for line in f:
            entity_id, k = map(int, line.rstrip().split(","))
            id2k_dict[entity_id] = k

    return id2k_dict

def load_k_values_sequence(data_name, sample, generator_name, args):
    id2k_dict = load_entity_id2k_dict(data_name, sample, generator_name, args)

    sequence = np.zeros(len(id2k_dict), dtype=int)

    sorted_entity_ids = sorted(list(id2k_dict.keys()))

    for idx, entity_id in enumerate(sorted_entity_ids):
        sequence[idx] = id2k_dict[entity_id]

    return sequence


def load_raw_clusters(data_name, sample, generator_name, info_loss_name, handler_name, calgo_name, args):
    clusters_path = putils.get_raw_clusters_path(data_name, sample, generator_name, info_loss_name, handler_name, calgo_name, args)
    clusters = algorithms.Clusters.from_file(clusters_path)

    logger.debug("loaded raw clusters at: {}".format(clusters_path))

    return clusters

def get_number_of_entities(data_name, sample):
    graph_dir = putils.get_raw_graph_path(data_name, sample)
    entities_idx_path = os.path.join(graph_dir, "entities.idx")

    num_entities = 0
    with open(entities_idx_path) as f:
        for _ in f:
            num_entities += 1

    return num_entities