from numpy.core import numeric
from sortedcontainers import SortedList
import itertools
import logging
import numpy as np

from anonygraph.info_loss.info import get_degree

logger = logging.getLogger(__name__)

def generate_adjency_matrix(graph, relation_id, id2idx):
    logger.debug("id2idx: {}".format(id2idx))
    # map idx to id
    adj_matrix = np.zeros(shape=(len(id2idx), len(id2idx)))

    cluster_entity_ids = list(id2idx.keys())

    for entity1_id, entity2_id in itertools.product(cluster_entity_ids, cluster_entity_ids):
        is_existed = graph.is_edge_existed(entity1_id, relation_id, entity2_id)
        logger.debug("({}, {}): {}".format(entity1_id, entity2_id, is_existed))

        if is_existed:
            entity1_idx = id2idx[entity1_id]
            entity2_idx = id2idx[entity2_id]

            adj_matrix[entity1_idx, entity2_idx] = 1

    return adj_matrix

def generate_id2idx_map(entity_ids):
    logger.debug("entity_ids: {}".format(entity_ids))
    id2idx = {}

    for idx, id in enumerate(entity_ids):
        id2idx[id] = idx

    return id2idx

def generate_clusters_matrix(clusters, id2idx):
    result = np.zeros(shape=(len(id2idx), len(id2idx)))

    for cluster in clusters:
        for entity1_id, entity2_id in itertools.product(cluster, cluster):
            entity1_idx = id2idx[entity1_id]
            entity2_idx = id2idx[entity2_id]

            result[entity1_idx, entity2_idx] = 1
            result[entity2_idx, entity1_idx] = 1

    return result

def generate_clusters_matrix2(clusters, id2idx):
    result = np.zeros(shape=(len(clusters), len(id2idx)))

    for cluster_idx, cluster in enumerate(clusters):
        for entity_id in cluster:
            entity_idx = id2idx[entity_id]

            result[cluster_idx, entity_idx] = 1

    return result

def generate_out_degree_array(adj_matrix):
    return np.sum(adj_matrix, axis=1)

def generate_in_degree_array(adj_matrix):
    return np.sum(adj_matrix, axis=0)

def get_expected_degree(clusters_matrix, degree_array):
    # logger.debug("clusters_matrix: {}".format(clusters_matrix))
    # logger.debug("degree_array: {}".format(degree_array))

    degree_matrix = clusters_matrix * degree_array
    # logger.debug("degree_matrix: {}".format(degree_matrix))
    max_degree_matrix = np.max(degree_matrix, axis=1,keepdims=True)
    max_cluster_matrix = max_degree_matrix * clusters_matrix
    max_cluster_degree_array = np.max(max_cluster_matrix, axis=0)
    expected_degree_array = max_cluster_degree_array - degree_array

    # logger.debug("degree matrix ({}): {}".format(degree_matrix.shape, degree_matrix))
    # logger.debug("max degree matrix ({}): {}".format(max_degree_matrix.shape, max_degree_matrix))
    # logger.debug("max cluster degree matrix ({}): {}".format(max_cluster_matrix.shape, max_cluster_matrix))
    # logger.debug("max_cluster_degree_array ({}): {}".format(max_cluster_degree_array.shape, max_cluster_degree_array))
    # logger.debug("expected_degree_array ({}): {}".format(expected_degree_array.shape, expected_degree_array))

    return expected_degree_array

def test_same_out_in_degree(adj_matrix, clusters_matrix):
    logger.debug("adj matrix: {}".format(adj_matrix))

    out_degree_array = generate_out_degree_array(adj_matrix)
    in_degree_array = generate_in_degree_array(adj_matrix)

    logger.debug("degree array out: {} \n in: {}".format(out_degree_array, in_degree_array))

    expected_out_degree_array = get_expected_degree(clusters_matrix, out_degree_array)
    expected_in_degree_array = get_expected_degree(clusters_matrix, in_degree_array)

    logger.debug("expected out degree array: {}".format(expected_out_degree_array))
    logger.debug("expected in degree array: {}".format(expected_in_degree_array))

    out_expected = np.sum(expected_out_degree_array)
    in_expected = np.sum(expected_in_degree_array)

    logger.debug("out expected: {} - in expected: {}".format(out_expected, in_expected))

    assert out_expected + in_expected == 0


def test_out_in_degree(out_degree_array, in_degree_array, subgraph, id2idx, entity_ids, relation_id):
    for entity_id in entity_ids:
        out_degree = get_degree(subgraph, entity_id, relation_id, "out")
        in_degree = get_degree(subgraph, entity_id, relation_id, "in")

        entity_idx = id2idx[entity_id]

        assert out_degree == out_degree_array[entity_idx], "expected: {} actual: {}".format(out_degree_array[entity_idx], out_degree)
        assert in_degree == in_degree_array[entity_idx], "expected: {} actual: {}".format(in_degree_array[entity_idx], in_degree)

def calculate_total_expected_degree(expected_out_degree_array, expected_in_degree_array):
    return np.sum(expected_out_degree_array) + np.sum(expected_in_degree_array)


def check_expected_degree(expected_out_degree_array, expected_in_degree_array):
    total_required = calculate_total_expected_degree(expected_out_degree_array, expected_in_degree_array)
    logger.debug("total required: {}".format(total_required))
    return total_required > 0

def get_sorted_expected_degree_array(expected_degree_array):
    sorted_expected_degree_array = SortedList(key=lambda item: -item[0])
    for entity_idx, expected_degree in enumerate(expected_degree_array):
        if expected_degree > 0:
            sorted_expected_degree_array.add((expected_degree, entity_idx))

    return sorted_expected_degree_array

def add_fake_edges_to_most_required(adj_matrix, out_degree_array, in_degree_array, expected_out_degree_array, expected_in_degree_array):
    num_entities = len(out_degree_array)
    logger.debug("num_entities: {}".format(num_entities))
    # raise Exception()
    total_required_out_degree = sum(expected_out_degree_array)
    total_required_in_degree = sum(expected_in_degree_array)
    logger.debug("init total_required out-/in-degree: {}, {}".format(total_required_out_degree, total_required_in_degree))

    num_fake_edges = 0

    sorted_expected_out_degree_array = get_sorted_expected_degree_array(expected_out_degree_array)
    sorted_expected_in_degree_array = get_sorted_expected_degree_array(expected_in_degree_array)

    logger.debug("sorted_expected_out_degree_array: {}".format(sorted_expected_out_degree_array))
    logger.debug("sorted_expected_in_degree_array: {}".format(sorted_expected_in_degree_array))

    sorted_out_idx = 0
    sorted_in_idx = 0
    while (total_required_in_degree > 0 and total_required_out_degree > 0):
        if len(sorted_expected_out_degree_array) * len(sorted_expected_in_degree_array) == 0:
            break

        logger.debug("sorted_expected_out/in_degree_array len: {}/{}".format(len(sorted_expected_out_degree_array), len(sorted_expected_in_degree_array)))

        logger.debug("sorted_expected_out_degree[{}]: {}".format(sorted_out_idx, sorted_expected_out_degree_array[sorted_out_idx]))
        logger.debug("sorted_expected_in_degree[{}]: {}".format(sorted_in_idx, sorted_expected_in_degree_array[sorted_in_idx]))

        # get the first, and check if he/she can be added out/in edges
        entity1_idx = sorted_expected_out_degree_array[sorted_out_idx][1]
        entity2_idx = sorted_expected_in_degree_array[sorted_in_idx][1]

        if adj_matrix[entity1_idx, entity2_idx] == 0:
            adj_matrix[entity1_idx, entity2_idx] = 1
            out_degree_array[entity1_idx] += 1
            in_degree_array[entity2_idx] += 1
            expected_out_degree_array[entity1_idx] -= 1
            expected_in_degree_array[entity2_idx] -= 1

            out_item = sorted_expected_out_degree_array.pop(sorted_out_idx)
            if out_item[0] > 1:
                sorted_expected_out_degree_array.add((out_item[0] - 1, out_item[1]))

            in_item = sorted_expected_in_degree_array.pop(sorted_in_idx)
            if in_item[0] > 1:
                sorted_expected_in_degree_array.add((in_item[0] - 1, in_item[1]))

            total_required_out_degree = total_required_out_degree - 1
            total_required_in_degree = total_required_in_degree - 1
            num_fake_edges += 1

            sorted_out_idx = 0
            sorted_in_idx = 0
            # logger.debug("add a fake edge")
        else:
            if expected_out_degree_array[entity1_idx] > expected_in_degree_array[entity2_idx] and sorted_out_idx < len(sorted_expected_out_degree_array) - 1:
                sorted_out_idx += 1
            else:
                sorted_in_idx += 1

                if sorted_in_idx == len(sorted_expected_in_degree_array):
                    break

            logger.debug("updated sorted-out/in-idx: {}/{}".format(sorted_out_idx, sorted_in_idx))

        # if expected_out_degree_array[entity1_idx] > 0 and expected_in_degree_array[entity2_idx] > 0 and adj_matrix[entity1_idx, entity2_idx] == 0:
        # logger.debug(sorted_expected_out_degree_array[0])
        # logger.debug(sorted_expected_in_degree_array[0])

    logger.debug("added {} fake edges".format(num_fake_edges))
    return num_fake_edges
    # raise Exception()

def add_fake_edges(adj_matrix, out_degree_array, in_degree_array, expected_out_degree_array, expected_in_degree_array):
    num_fake_edges = 0
    step = 0
    num_entities = len(out_degree_array)
    logger.debug("num_entities: {}".format(num_entities))
    # raise Exception()
    total_required_out_degree = sum(expected_out_degree_array)
    total_required_in_degree = sum(expected_in_degree_array)
    logger.debug("init total_required out-/in-degree: {}, {}".format(total_required_out_degree, total_required_in_degree))
    # raise Exception()
    entity1_idx = 0
    entity2_idx = 0

    while(total_required_out_degree > 0 and total_required_in_degree > 0):
        # logger.debug("entity1_idx, entity2_idx: {}, {}".format(entity1_idx, entity2_idx))

        if expected_out_degree_array[entity1_idx] > 0 and expected_in_degree_array[entity2_idx] > 0 and adj_matrix[entity1_idx, entity2_idx] == 0:
            adj_matrix[entity1_idx, entity2_idx] = 1
            out_degree_array[entity1_idx] += 1
            in_degree_array[entity2_idx] += 1
            expected_out_degree_array[entity1_idx] -= 1
            expected_in_degree_array[entity2_idx] -= 1

            total_required_out_degree = total_required_out_degree - 1
            total_required_in_degree = total_required_in_degree - 1
            num_fake_edges += 1

            # logger.debug("add a fake edge")

            # logger.debug("required out-/in-degree: {}/{}".format(total_required_out_degree, total_required_in_degree))
            # logger.debug("max required out degree: {}".format(max(expected_out_degree_array)))
            # logger.debug("max required in degree: {}".format(max(expected_in_degree_array)))

        entity2_idx += 1

        if entity2_idx >= num_entities:
            entity1_idx += 1
            entity2_idx = 0

        if entity1_idx >= num_entities:
            break




    # while(check_expected_degree(expected_out_degree_array, expected_in_degree_array)):
    #     num_step_fake_edges = 0
    #     for entity1_idx, entity2_idx in itertools.permutations(range(num_entities), r=2):
    #         if expected_out_degree_array[entity1_idx] > 0 and expected_in_degree_array[entity2_idx] > 0 and adj_matrix[entity1_idx, entity2_idx] == 0:
    #             adj_matrix[entity1_idx, entity2_idx] = 1
    #             out_degree_array[entity1_idx] += 1
    #             in_degree_array[entity2_idx] += 1
    #             expected_out_degree_array[entity1_idx] -= 1
    #             expected_in_degree_array[entity2_idx] -= 1

    #             num_step_fake_edges += 1

    #     logger.debug("[step: {}] added {} fake edges".format(step, num_step_fake_edges))

    #     if num_step_fake_edges == 0:
    #         break


    #     step += 1
    #     num_fake_edges += num_step_fake_edges

    logger.debug("expected_out_degree: {}".format(sum(expected_out_degree_array)))
    logger.debug("expected_in_degree: {}".format(sum(expected_in_degree_array)))
    logger.debug("added {} fake edges in {} steps".format(num_fake_edges, step))

    # if num_fake_edges < 100:
    #     raise Exception()
    return num_fake_edges
    # raise Exception()

def remove_randomly_an_out_edge(adj_matrix, entity_idx, out_degree_array, in_degree_array):
    random_idx = np.random.randint(adj_matrix.shape[0])
    logger.debug("start removing an out edge of {}".format(entity_idx))
    logger.debug("out edges: {}".format(adj_matrix[entity_idx, :]))

    while(adj_matrix[entity_idx, random_idx] == 0):
        random_idx = np.random.randint(adj_matrix.shape[0])

    logger.debug("remove out edge: {} ({})".format((entity_idx, random_idx), adj_matrix[entity_idx, random_idx]))

    adj_matrix[entity_idx, random_idx] = 0
    out_degree_array[entity_idx] -= 1
    in_degree_array[random_idx] -= 1


def remove_randomly_an_in_edge(adj_matrix, entity_idx, out_degree_array, in_degree_array):
    random_idx = np.random.randint(adj_matrix.shape[0])
    logger.debug("start removing an in edge of {}".format(entity_idx))
    logger.debug("in edges: {}".format(adj_matrix[:, entity_idx]))

    while(adj_matrix[random_idx, entity_idx] == 0):
        random_idx = np.random.randint(adj_matrix.shape[0])

    logger.debug("remove in edge: {} ({})".format((random_idx, entity_idx), adj_matrix[random_idx, entity_idx]))

    adj_matrix[random_idx, entity_idx] = 0
    out_degree_array[random_idx] -= 1
    in_degree_array[entity_idx] -= 1

def get_max_out_in_degree(out_degree_array, in_degree_array):
    max_out_degree = np.max(out_degree_array)
    max_in_degree = np.max(in_degree_array)

    return max_out_degree, max_in_degree

def decrease_max_required_degree(adj_matrix, clusters_matrix, out_degree_array, in_degree_array, expected_out_degree_array, expected_in_degree_array):
    # find maximum required out-/in-degree

    sorted_expected_out_degree_array = get_sorted_expected_degree_array(expected_out_degree_array)
    sorted_expected_in_degree_array = get_sorted_expected_degree_array(expected_in_degree_array)

    logger.debug("sorted_expected_out_degree_array (len: {}): {}".format(len(sorted_expected_out_degree_array), sorted_expected_out_degree_array))
    logger.debug("sorted_expected_in_degree_array (len: {}): {}".format(len(sorted_expected_in_degree_array), sorted_expected_in_degree_array))

    if len(sorted_expected_out_degree_array) > 0:
        highest_expected_out_degree, highest_expected_out_entity_idx = sorted_expected_out_degree_array[0]
    else:
        highest_expected_out_degree = 0

    if len(sorted_expected_in_degree_array) > 0:
        highest_expected_in_degree, highest_expected_in_entity_idx = sorted_expected_in_degree_array[0]
    else:
        highest_expected_in_degree = 0

    num_clusters = clusters_matrix.shape[0]
    logger.debug("highest_expected_out/in_degree: {}/{}".format(highest_expected_out_degree, highest_expected_in_degree))
    logger.debug("clusters_matrix: {}".format(clusters_matrix.shape))

    if highest_expected_out_degree == 0 and highest_expected_in_degree == 0:
        logger.debug("do not remove any edge")

    if highest_expected_out_degree > highest_expected_in_degree:
        # decrease max expected out degree
        logger.debug("decrease max expected out degree")

        # find clusters that have highest expected out degree
        max_expected_degree_cluster_idx = find_highest_expected_degree_cluster(clusters_matrix, sorted_expected_out_degree_array)


        # remove max out degree of that cluster
        degree_max_degree_cluster(adj_matrix, clusters_matrix, out_degree_array, in_degree_array, max_expected_degree_cluster_idx, "out")

    else:
        # decrease max expected in degree
        logger.debug("decrease max expected in degree")

        # find clusters that have highest expected out degree
        max_expected_degree_cluster_idx = find_highest_expected_degree_cluster(clusters_matrix, sorted_expected_in_degree_array)

        # decrease degree
        degree_max_degree_cluster(adj_matrix, clusters_matrix, out_degree_array, in_degree_array, max_expected_degree_cluster_idx, "in")



    # raise Exception()

def check_adj_degree_matrix(adj_matrix, out_degree_array, in_degree_array):
    num_entities = adj_matrix.shape[0]

    for entity_idx in range(num_entities):
        out_degree = np.sum(adj_matrix[entity_idx, :])
        in_degree = np.sum(adj_matrix[:, entity_idx])

        assert out_degree_array[entity_idx] == out_degree and in_degree_array[entity_idx] == in_degree, "Entity_idx: {} - Out-degree and in-degree calculated from adj_matrix ({},{}) is different from out/in-degree array ({},{})".format(entity_idx, out_degree, in_degree, out_degree_array[entity_idx], in_degree_array[entity_idx])


def calculate_degree(out_degree_array, in_degree_array, entity_idx, degree_type):
    if degree_type == "out":
        degree = out_degree_array[entity_idx]
    elif degree_type == "in":
        degree = in_degree_array[entity_idx]
    else:
        raise Exception("Unsupported degree type: {}".format(degree_type))
    return degree

def find_connected_entity_idxes(adj_matrix, entity_idx, degree_type):
    if degree_type == "out":
        connecting_entity_idxes = np.where(adj_matrix[entity_idx, :] == 1)[0]
    elif degree_type == "in":
        connecting_entity_idxes = np.where(adj_matrix[:, entity_idx] == 1)[0]
    else:
        raise Exception("Unsupported degree type: {}".format(degree_type))

    return connecting_entity_idxes

def remove_edge(adj_matrix, out_degree_array, in_degree_array, entity_idx, connected_entity_idx, degree_type):
    if degree_type == "out":
        adj_matrix[entity_idx, connected_entity_idx] = 0
        out_degree_array[entity_idx] -= 1
        in_degree_array[connected_entity_idx] -= 1
    elif degree_type == "in":
        adj_matrix[connected_entity_idx, entity_idx] = 0
        in_degree_array[entity_idx] -= 1
        out_degree_array[connected_entity_idx] -= 1
    else:
        raise Exception("Unsupported degree type: {}".format(degree_type))

def degree_max_degree_cluster(adj_matrix, clusters_matrix, out_degree_array, in_degree_array, cluster_idx, degree_type):
    # find entity idxes of the cluster
    entity_idxes = np.where(clusters_matrix[cluster_idx,:]==1)[0]
    logger.debug("entity idxes in cluster {}: {}".format(cluster_idx, entity_idxes))

    # find entity idx that have highest out degree
    sorted_entity_degree = SortedList(key=lambda item: -item[1])
    for entity_idx in entity_idxes:
        degree = calculate_degree(out_degree_array, in_degree_array, entity_idx, degree_type)
        # degree = degree_array[entity_idx]
        # degree = calculate_degree(adj_matrix, entity_idx, degree_type)
        sorted_entity_degree.add((entity_idx, degree))


        # in_degree = np.sum(adj_matrix[:, entity_idx])

        logger.debug("entity idx: {} - degree: {}".format(entity_idx, degree))
        # sorted_entity_degree.add((entity_idx, in_degree))


    logger.debug("sorted_entity_degree: {}".format(sorted_entity_degree))

    # choose in-edges to remove without affecting other edges
    max_degree = sorted_entity_degree[0][1]
    logger.debug("max_degree: {}".format(max_degree))
    # raise Exception()
    for entity_idx, degree in sorted_entity_degree:
        if degree == max_degree:
            # connecting_entity_idxes = np.where(adj_matrix[:, entity_idx] == 1)[0]
            connecting_entity_idxes = find_connected_entity_idxes(adj_matrix, entity_idx, degree_type)
            logger.debug("connecting_entity_idxes (len: {}): {}".format(len(connecting_entity_idxes), connecting_entity_idxes))

            # remove those in smallest cluster
            sorted_entity_cluster_size = SortedList(key=lambda item: item[1])
            for connecting_entity_idx in connecting_entity_idxes:
                cluster_idx = np.where(clusters_matrix[:,connecting_entity_idx] == 1)[0][0]

                cluster_size = np.sum(clusters_matrix[cluster_idx])
                sorted_entity_cluster_size.add((connecting_entity_idx, cluster_size))

            logger.debug("sorted_entity_cluster_size: {}".format(sorted_entity_cluster_size))

            smallest_cluster_size_connecting_entity_idx = sorted_entity_cluster_size[0][0]

            remove_edge(adj_matrix, out_degree_array, in_degree_array, entity_idx,smallest_cluster_size_connecting_entity_idx, degree_type)
            # adj_matrix[smallest_cluster_size_connecting_entity_idx, entity_idx] = 0
        else:
            break


def find_highest_expected_degree_cluster(clusters_matrix, sorted_expected_degree_array):
    cluster_idx2expected_degree = {}

    for expected_degree, entity_idx in sorted_expected_degree_array:
        entity_cluster_idx = np.where(clusters_matrix[:,entity_idx] == 1)[0][0]

        cluster_idx2expected_degree[entity_cluster_idx] = cluster_idx2expected_degree.get(entity_cluster_idx, 0) + expected_degree
        # logger.debug(entity_cluster_idx)

        logger.debug("entity idx: {} - cluster idx: {} - expected degree: {}".format(entity_idx, entity_cluster_idx, expected_degree))


    logger.debug(cluster_idx2expected_degree)
    max_expected_degree_item = (-1, 0)
    for cluster_idx, cluster_freq in cluster_idx2expected_degree.items():
        if cluster_freq > max_expected_degree_item[1]:
            max_expected_degree_item = (cluster_idx, cluster_freq)

    logger.debug("cluster_idx: {} - max_expected_degree: {}".format(max_expected_degree_item[0], max_expected_degree_item[1]))
    return max_expected_degree_item[0]

def decrease_max_degree(adj_matrix, out_degree_array, in_degree_array):
    max_out_degree, max_in_degree = get_max_out_in_degree(out_degree_array, in_degree_array)

    # logger.debug("adj matrix: {}".format(adj_matrix))
    # logger.debug("array of out: {} \n in: {}".format(out_degree_array, in_degree_array))
    # logger.debug("max out: {} - in: {}".format(max_out_degree, max_in_degree))

    num_removed_edges = 0
    if max_out_degree > max_in_degree:
        logger.info("removing to decrease out-degree")
        # entity_idx = np.argmax(out_degree_array)
        for entity_idx in range(len(out_degree_array)):
            if out_degree_array[entity_idx] == max_out_degree:
                remove_randomly_an_out_edge(adj_matrix, entity_idx, out_degree_array, in_degree_array)
                num_removed_edges += 1

                # logger.info("max out/in-degree: {}".format(get_max_out_in_degree(out_degree_array, in_degree_array)))
    else:
        logger.info("removing to decrease in-degree")
        for entity_idx in range(len(out_degree_array)):
            if in_degree_array[entity_idx] == max_in_degree:
        # entity_idx = np.argmax(in_degree_array)
                remove_randomly_an_in_edge(adj_matrix, entity_idx, out_degree_array, in_degree_array)
                num_removed_edges += 1
                logger.debug("max out/in-degree: {}".format(get_max_out_in_degree(out_degree_array, in_degree_array)))

    logger.info("final max out/in-degree: {}".format(get_max_out_in_degree(out_degree_array, in_degree_array)))
    logger.info("removed {} edges".format(num_removed_edges))

def apply_adj_matrix_to_add_or_remove(adj_matrix, subgraph, entity_ids, relation_id, entity1_idx, entity2_idx):
    entity1_id = entity_ids[entity1_idx]
    entity2_id = entity_ids[entity2_idx]

    # logger.debug("{} - {}".format(entity1_idx, entity2_idx))

    if adj_matrix[entity1_idx, entity2_idx] == 1:
        if subgraph.is_edge_existed(entity1_id, relation_id, entity2_id):
            # do nothing
            pass
        else:
            # add fake edge
            subgraph.add_relationship_edge_from_id(entity1_id, relation_id, entity2_id)
    else:
        if subgraph.is_edge_existed(entity1_id, relation_id, entity2_id):
            # remove an edge
            subgraph.remove_edge_from_id(entity1_id, relation_id, entity2_id)
        else:
            # do nothing
            pass

def apply_adj_matrix_to_subgraph(adj_matrix, subgraph, entity_ids, relation_id):
    for entity1_idx, entity2_idx in itertools.combinations(range(adj_matrix.shape[0]), r=2):
        apply_adj_matrix_to_add_or_remove(adj_matrix, subgraph, entity_ids, relation_id, entity1_idx, entity2_idx)
        apply_adj_matrix_to_add_or_remove(adj_matrix, subgraph, entity_ids, relation_id, entity2_idx, entity1_idx)


def calculate_expected_out_in_degree_array(clusters_matrix, out_degree_array, in_degree_array):
    expected_out_degree_array = get_expected_degree(clusters_matrix, out_degree_array)
    expected_in_degree_array = get_expected_degree(clusters_matrix, in_degree_array)

    return expected_out_degree_array, expected_in_degree_array



class SameDegreeRelationshipsGenerator:
    def __init__(self):
        pass

    def __call__(self, graph, clusters):
        logger.info("start generalizing with degree")
        logger.debug("clusters: {}".format(clusters))
        # keep adding

        # entity_ids = list(subgraph.entity_ids)
        clusters_entity_ids = []
        for cluster in clusters:
            clusters_entity_ids.extend(cluster)

        logger.debug("clusters_entity_ids: {}".format(clusters_entity_ids))

        id2idx = generate_id2idx_map(clusters_entity_ids)
        logger.debug("id2idx: {}".format(id2idx))

        clusters_matrix = generate_clusters_matrix2(clusters, id2idx)
        # logger.debug("clusters matrix: {}".format(clusters_matrix))


        # for every relation_id, generate
        # anonymity = gmetrics.calculate_anonymity("out_in")(subgraph, subgraph, None)
        # logger.debug("out-/in-degree anonymity: {}".format(anonymity))

        for relation_id in graph.relationship_relation_ids:
            logger.info("generalizing relation_id: {}".format(relation_id))

            adj_matrix = generate_adjency_matrix(graph, relation_id, id2idx)

            out_degree_array = generate_out_degree_array(adj_matrix)
            in_degree_array = generate_in_degree_array(adj_matrix)

            # logger.debug("adj matrix ({}): {}".format(adj_matrix.shape, adj_matrix))
            # logger.debug("out degree: {} \n in degree: {}".format(out_degree_array, in_degree_array))

            expected_out_degree_array, expected_in_degree_array = calculate_expected_out_in_degree_array(clusters_matrix, out_degree_array, in_degree_array)

            total_expected_degree = calculate_total_expected_degree(expected_out_degree_array, expected_in_degree_array)

            logger.debug("expected out degree: {} \n in degree: {}".format(expected_out_degree_array, expected_in_degree_array))
            logger.debug("total expected {} degree".format(total_expected_degree))
            logger.info("init total_expected_degree: {}".format(total_expected_degree))

            current_total_expected_degree = total_expected_degree
            previous_total_expected_degree = current_total_expected_degree

            check_adj_degree_matrix(adj_matrix, out_degree_array, in_degree_array)
            while(total_expected_degree > 0):
                # keep adding
                logger.info("current total_expected_degree: {}".format(total_expected_degree))
                logger.info("adding fake edges")
                num_added_fake_edges = add_fake_edges_to_most_required(adj_matrix, out_degree_array, in_degree_array, expected_out_degree_array, expected_in_degree_array)
                logger.info("added {} fake edges".format(num_added_fake_edges))
                check_adj_degree_matrix(adj_matrix, out_degree_array, in_degree_array)

                expected_out_degree_array, expected_in_degree_array = calculate_expected_out_in_degree_array(clusters_matrix, out_degree_array, in_degree_array)

                total_expected_degree = calculate_total_expected_degree(expected_out_degree_array, expected_in_degree_array)
                check_adj_degree_matrix(adj_matrix, out_degree_array, in_degree_array)
                logger.info("total_expected_degree after adding fake edges: {}".format(total_expected_degree))
                if total_expected_degree == 0:
                    break

                logger.info("decreasing max degree")
                decrease_max_required_degree(adj_matrix, clusters_matrix, out_degree_array, in_degree_array, expected_out_degree_array, expected_in_degree_array)

                check_adj_degree_matrix(adj_matrix, out_degree_array, in_degree_array)
                expected_out_degree_array, expected_in_degree_array = calculate_expected_out_in_degree_array(clusters_matrix, out_degree_array, in_degree_array)

                total_expected_degree = calculate_total_expected_degree(expected_out_degree_array, expected_in_degree_array)
                logger.info("total_expected_degree after removing: {}".format(total_expected_degree))

                logger.info("num remaining edges: {}".format(np.sum(adj_matrix)))

            logger.info("finished generalizing relation_id: {}".format(relation_id))

            test_same_out_in_degree(adj_matrix, clusters_matrix)

            apply_adj_matrix_to_subgraph(adj_matrix, graph, clusters_entity_ids, relation_id)


        # anonymity = gmetrics.calculate_anonymity("out_in")(subgraph, subgraph, None)
        # logger.info("out-/in-degree anonymity: {}".format(anonymity))
        # raise Exception()