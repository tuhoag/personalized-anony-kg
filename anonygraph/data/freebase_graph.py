from tqdm import tqdm
import logging
import os

from .static_graph import StaticGraph
# from .freebase_graph import load_static_graph_from_old_output

logger = logging.getLogger(__name__)


def get_name(prefix, id):
    return prefix + '_' + id

def get_user_name(id):
    return get_name('user', id)


class FreebaseGraph(StaticGraph):
    @staticmethod
    def from_raw_file(data_dir, args):
        graph = FreebaseGraph()
        load_static_graph_from_old_output(data_dir, graph, "user", "", args)

        return graph


def load_edges_from_old_output(graph, file_path, edge_type, relation_name_dict, node_name_dict):
    edge_addition_fn_dict = {
        "attr": graph.add_attribute_edge,
        "rel": graph.add_relationship_edge,
    }

    edge_addition_fn = edge_addition_fn_dict[edge_type]

    with open(file_path, "r") as f:
        for line in f:
            node1_id, relation_id, node2_id = map(int, line.strip().split(","))
            node1_name = node_name_dict[node1_id]
            node2_name = node_name_dict[node2_id]
            relation_name = relation_name_dict[relation_id]

            edge_addition_fn(node1_name, relation_name, node2_name)

def read_index_files(name_prefix, file_paths):
    result = {}
    for path in file_paths:
        logger.debug("loadding from: {}".format(path))
        with open(path, "r") as f:
            for line in f:
                logger.debug(line)
                splits = line.strip().split(",")
                name = ",".join(splits[0:-1])
                id = int(splits[-1])
                if id not in result:
                    result[id] = "{}_{}".format(name_prefix, name)
                else:
                    logger.debug(result)
                    raise Exception("Duplicate id: {}".format(id))

    return result

def load_static_graph_from_old_output(data_dir, graph, node_name_prefix, relation_name_prefix, args):
    node_name_dict = read_index_files(node_name_prefix, [
        os.path.join(data_dir, "users.idx"),
        os.path.join(data_dir, "values.idx")
    ])
    relation_name_dict = read_index_files(relation_name_prefix, [
        os.path.join(data_dir, "attrs.idx"),
        os.path.join(data_dir, "rels.idx")
    ])
    logger.debug(node_name_dict)
    logger.debug(relation_name_dict)

    attr_edges_path = os.path.join(data_dir, "attrs.edges")
    load_edges_from_old_output(graph, attr_edges_path, "attr", relation_name_dict, node_name_dict)
    rel_edges_path = os.path.join(data_dir, "rels.edges")
    load_edges_from_old_output(graph, rel_edges_path, "rel", relation_name_dict, node_name_dict)