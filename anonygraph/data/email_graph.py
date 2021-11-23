import networkx as nx
import logging
import os

from .static_graph import StaticGraph
# from .freebase_graph import load_static_graph_from_old_output

logger = logging.getLogger(__name__)

GRAPH_NAME = 'email'
USER_RELATION_NAME = 'sent'
ATTR_RELATION_NAME = 'belongs_to'

def get_name(prefix, id):
    return prefix + '_' + id

def get_user_name(id):
    return get_name('user', id)

def get_dept_name(id):
    return get_name('dept', id)

def load_users_relationship(graph, file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.rstrip().split(' '), lines))

        for u, v in lines:
            graph.add_relationship_edge(get_user_name(u), USER_RELATION_NAME, get_user_name(v))

def load_users_departments(graph, file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.rstrip().split(' '), lines))

        for u, v in lines:
            graph.add_attribute_edge(get_user_name(u), ATTR_RELATION_NAME, get_dept_name(v))

class EmailGraph(StaticGraph):
    @staticmethod
    def from_raw_file(data_dir, args):
        graph = EmailGraph()

        user_department_path = os.path.join(
            data_dir, 'email-Eu-core-department-labels.txt')
        load_users_departments(graph, user_department_path)

        user_relationship_path = os.path.join(data_dir, 'email-Eu-core.txt')
        load_users_relationship(graph, user_relationship_path)

        return graph
