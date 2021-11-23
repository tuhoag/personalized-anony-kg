import networkx as nx
import logging
import os


from .static_graph import StaticGraph

logger = logging.getLogger(__name__)

class DummyGraph(StaticGraph):

    @staticmethod
    def from_raw_file(data_dir, args):
        # load attributes
        # load relationships
        graph = StaticGraph()

        graph.add_attribute_edge('user_0', 'age', 21)
        graph.add_attribute_edge('user_0', 'job', 'Student')

        graph.add_attribute_edge('user_1', 'age', 19)
        graph.add_attribute_edge('user_1', 'job', 'Student')

        graph.add_attribute_edge('user_2', 'age', 21)
        graph.add_attribute_edge('user_2', 'job', 'Engineer')

        graph.add_attribute_edge('user_3', 'age', 30)
        graph.add_attribute_edge('user_3', 'job', 'Engineer')

        graph.add_attribute_edge('user_4', 'age', 30)
        graph.add_attribute_edge('user_4', 'job', 'Engineer')

        graph.add_relationship_edge('user_0', 'follow', 'user_2')

        return graph
