import anonygraph
import pygraphviz as pgv
import networkx as nx
import matplotlib.pyplot as plt

users_data = {
    "age": {
        21: ["Ken", "Henry"],
        19: ["Mary", "Jane"],
        30: ["Tom"],
    },
    "job": {
        "Student": ["Ken", "Mary", "Jane"],
        "Engineer": ["Henry", "Tom"],
    },
    "follow": {
        "Henry": ["Ken"],
    },
    "classmate of": {
        "Jane": ["Mary"]
    }
}

name2k_dict = {
    "Ken": 2,
    "Henry": 2,
    "Tom": 1,
    "Mary": 2,
    "Jane": 4,
}

user_names = ["Ken", "Mary", "Henry", "Tom", "Jane"]

anony_users_data = {
    "age": {
        21: user_names,
        19: user_names,
        30: user_names,
    },
    "job": {
        "Student": user_names,
        "Engineer": user_names,
    },
    "follow": {
        "Henry": ["Ken"],
        "Tom": ["Henry"],
        "Mary": ["Tom"],
        "Jane": ["Mary"],
        "Ken": ["Jane"],
    },
    "classmate of": {
        "Henry": ["Ken"],
        "Tom": ["Henry"],
        "Mary": ["Tom"],
        "Jane": ["Mary"],
        "Ken": ["Jane"],
    }
}

new_anony_users_data = {
    "age": {
        21: ["Mary", "Ken", "Henry", "Tom"],
        19: ["Mary", "Ken"],
        30: ["Henry", "Tom"],
    },
    "job": {
        "Student": ["Ken", "Mary"],
        "Engineer": ["Henry", "Tom"],
    },
    "follow": {
        "Henry": ["Ken"],
        "Tom": ["Mary"],
    },
}

new_users_data = {
    "age": {
        21: ["Ken", "Henry"],
        19: ["Mary"],
        30: ["Tom"],
    },
    "job": {
        "Student": ["Ken", "Mary"],
        "Engineer": ["Henry", "Tom"],
    },
    "follow": {
        "Henry": ["Ken"],
    },
}

attr_relations = ["age", "job"]

def is_attr_relation(relation_name):
    return relation_name in attr_relations

def add_edge(graph, start_node, end_node, relation, anony_mode, line_style, lib):
    start_node_name = get_node_name(start_node, False, anony_mode)
    end_node_name = get_node_name(end_node, is_attr_relation(relation), anony_mode)

    print(start_node_name, relation, end_node_name)
    if lib == "pygraphviz":
        graph.add_edge(start_node_name, end_node_name, key=relation, label = relation, style=line_style)
    elif lib == "networkx":
        graph.add_edge(start_node_name, end_node_name, relation, label=relation)
    else:
        raise Exception()

def get_node_name(node_name, is_value, anony_mode):
    if is_value:
        return node_name

    if anony_mode == "raw":
        return "{}-{}".format(node_name, name2k_dict[node_name])
    elif anony_mode == "anony":
        return "user:{}".format(user_names.index(node_name))
    else:
        raise Exception()



def generate_graph(g, users_data, line_style, lib="pygraphviz"):
    for relation, relation_data in users_data.items():
        for end_node, start_nodes in relation_data.items():
            for start_node in start_nodes:
                add_edge(g, start_node, end_node, relation, "raw", line_style, lib)
                # g.add_edge(start_node, end_node, relation, label = relation, style=line_style)


def generate_anony_graph(g, anony_users_data, line_style, lib="pygraphviz"):
    for relation, relation_data in anony_users_data.items():
        for end_node, start_nodes in relation_data.items():
            for start_node in start_nodes:
                if not g.has_edge(start_node, end_node, relation):
                    add_edge(g, start_node, end_node, relation, "anony", line_style, lib)
                    # g.add_edge(start_node, end_node, relation, label = relation, style=line_style)

def visualize_with_pygraphviz(users_data, anony_users_data):
    graph = pgv.AGraph(directed=True)

    generate_graph(graph, users_data, "solid")
    # generate_anony_graph(graph, anony_users_data, "dashed")

    print(graph.string())  # print to screen
    graph.layout("circo")
    # graph.layout("dot")
    graph.draw("simple.png")
    # A.write("simple.dot")  # write to simple.dot

    # B = pgv.AGraph("simple.dot")  # create a new graph from file

    # B.layout()  # layout with default (neato)
    # B.draw("simple.png")  # draw png

def visualize_with_networkx(users_data, anony_users_data):
    graph = nx.MultiDiGraph()

    generate_graph(graph, users_data, "solid", "networkx")

    # for user_name, user_data in users_data.items():
    #     for relation, val in user_data.items():
    #         graph.add_edge(user_name, val, label=relation)

    pos = nx.layout.planar_layout(graph)
    # pos = nx.bipartite_layout(G, top)
    nx.draw_networkx(graph, with_label = True, pos=pos)
    plt.show()


def visualize_raw_graph(users_data):
    graph = pgv.AGraph(directed=True)

    generate_graph(graph, users_data, "solid")

    print(graph.string())  # print to screen
    graph.layout("dot")
    # graph.layout("dot")
    graph.draw("graph_raw.pdf")

def visualize_anony_graph(users_data, anony_users_data):
    graph = pgv.AGraph(directed=True)

    generate_anony_graph(graph, users_data, "solid")
    generate_anony_graph(graph, anony_users_data, "dashed")

    print(graph.string())  # print to screen
    # graph.layout("circo")
    graph.layout("dot")
    graph.draw("graph_anony.pdf")

visualize_raw_graph(users_data)
visualize_anony_graph(new_users_data, new_anony_users_data)
# visualize_with_networkx(users_data, anony_users_data)