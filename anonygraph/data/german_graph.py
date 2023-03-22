import os

from anonygraph.data.static_graph import StaticGraph

def load_attributes(graph, file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.rstrip().split(' '), lines))

        for row_id, row in enumerate(lines):
            for attr_id, value in enumerate(row):
                attr_name = "attr_{}".format(attr_id)
                user_name = "user_{}".format(row_id)
                graph.add_attribute_edge(user_name, attr_name, value)


class GermanCreditGraph(StaticGraph):
    @staticmethod
    def from_raw_file(data_dir, args):
        graph = GermanCreditGraph()

        attributes_path = os.path.join(
            data_dir, 'german.data')
        load_attributes(graph, attributes_path)

        return graph
