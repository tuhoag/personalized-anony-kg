import logging

import anonygraph.info_loss as ifn
from anonygraph.constants import *
import anonygraph.utils.data as dutils

logger = logging.getLogger(__name__)

def calculate_remaining_adm(clusters, graph, args):
    if len(clusters) == 0:
        return 1

    ifn_fn = ifn.get_info_loss_function(args["info_loss"], graph, args)
    # ifn_fn = ifn.AttributeOutInDegreeInfoLoss(graph, {"alpha_adm": 0.5, "alpha_dm": 0.5})

    entity_ids_set = graph.entity_ids
    clusters_entity_ids_set = set()

    result = 0
    for cluster in clusters:
        result += ifn_fn.call(cluster.to_list()) * cluster.num_entities
        clusters_entity_ids_set.update(cluster.entity_ids)

    result = result / len(entity_ids_set)

    return result

def calculate_adm(clusters, graph, args):
    if len(clusters) == 0:
        return 1

    # raise Exception("args: {}".format(args))
    ifn_fn = ifn.get_info_loss_function(args["info_loss"], graph, args)

    entity_ids_set = graph.entity_ids
    clusters_entity_ids_set = set()

    result = 0
    for cluster in clusters:
        result += ifn_fn.call(cluster.to_list()) * cluster.num_entities
        clusters_entity_ids_set.update(cluster.entity_ids)

    removed_entity_ids = entity_ids_set.difference(clusters_entity_ids_set)

    result += len(removed_entity_ids) * 1

    result = result / len(entity_ids_set)

    return result

def calculate_num_fake_entities(clusters, graph, args):
    count = 0

    for cluster in clusters:
        for entity_id in cluster:
            if not graph.is_entity_id(entity_id):
                count += 1

    return count

def calculate_num_real_entities(clusters, graph, args):
    count = 0

    for cluster in clusters:
        for entity_id in cluster:
            if graph.is_entity_id(entity_id):
                count += 1

    return count

def calculate_num_real_edges(clusters, graph, args):
    return graph.num_edges

def calculate_num_fake_edges(clusters, graph, args):
    result = 0
    relation_ids = graph.relationship_relation_ids
    logger.debug("relation ids: {}".format(relation_ids))
    for cluster in clusters:
        out_union_info, out_entities_info = ifn.info.get_generalized_degree_info(graph, cluster, "out")
        in_union_info, in_entities_info = ifn.info.get_generalized_degree_info(graph, cluster, "in")

        for relation_id in relation_ids:
            out_generalized_degree = out_union_info.get(relation_id, 0)
            in_generalized_degree = in_union_info.get(relation_id, 0)

            for entity_id in cluster:
                out_entity_degree = out_entities_info.get(entity_id).get(relation_id, 0)
                in_entity_degree = in_entities_info.get(entity_id).get(relation_id, 0)

                logger.debug(out_entities_info)
                logger.debug(in_entities_info)
                # raise Exception(relation_id)

                # if out_generalized_degree is None or out_entity_degree is None:
                    # raise Exception("{}({}) {} {} {}".format(relation_id, relation_ids, out_union_info, out_generalized_degree, out_entity_degree))
                    # raise Exception("{} {} {} {}".format(out_union_info, out_entities_info, in_union_info, in_entities_info))
                out_degree_dif = out_generalized_degree - out_entity_degree
                in_degree_dif = in_generalized_degree - in_entity_degree

                generalized_degree = min(out_degree_dif, in_degree_dif)
                result += generalized_degree

    return result

def calculate_dm(clusters, graph, args):
    if len(clusters) == 0:
        return 1

    ifn_fn = ifn.get_info_loss_function(args["info_loss"], graph, args)

    result = 0
    for cluster in clusters:
        result += ifn_fn.call(cluster.to_list())

    result = result / len(clusters)

    return result

def calculate_odm(clusters, graph, args):
    if len(clusters) == 0:
        return 1

    ifn_fn = ifn.get_info_loss_function(args["info_loss"], graph, args)

    result = 0
    for cluster in clusters:
        result += ifn_fn.call(cluster.to_list())

    result = result / len(clusters)

    return result

def calculate_idm(clusters, graph, args):
    if len(clusters) == 0:
        return 1

    ifn_fn = ifn.get_info_loss_function(args["info_loss"], graph, args)

    result = 0
    for cluster in clusters:
        result += ifn_fn.call(cluster.to_list())

    result = result / len(clusters)

    return result

def calculate_am(clusters, graph, args):
    if len(clusters) == 0:
        return 1

    ifn_fn = ifn.get_info_loss_function(args["info_loss"], graph, args)

    result = 0
    for cluster in clusters:
        result += ifn_fn.call(cluster.to_list())

    result = result / len(clusters)

    return result

def calculate_anonymity(clusters, graph, args):
    if len(clusters) == 0:
        return 0

    clusters_sizes = [len(cluster) for cluster in clusters]
    anonymity = min(clusters_sizes)

    # raise Exception("clusters: {} - anonymity: {} ({})".format(clusters, anonymity, clusters_sizes))

    return anonymity

def calculate_removed_entities(clusters, graph, args):
    clusters_entity_ids = set()

    for cluster in clusters:
        clusters_entity_ids.update(cluster.entity_ids)

    removed_entity_ids = graph.entity_ids.difference(clusters_entity_ids)
    return len(removed_entity_ids)

def calculate_raw_entities(clusters, graph, args):
    return graph.num_entities

def calculate_num_invalid_anonymity_clusters(clusters, graph, args):
    #   load k values of that clusters
    id2k_dict = dutils.load_entity_id2k_dict(args["data"], args["sample"], args["gen"], args)
    logger.debug("loaded id2k: {}".format(id2k_dict))

    count = 0

    #   for each cluster
    for cluster in clusters:
    #       calculate max k
        k_values = list(map(lambda entity_id: id2k_dict[entity_id], cluster))
        max_k = max(k_values)

        logger.debug("max k values: {} among {}".format(max_k, k_values))

        #       if max k > num entities of cluster
        if max_k > cluster.num_entities:
            count += 1

    return count

def calculate_num_big_clusters(clusters, graph, args):
    id2k_dict = dutils.load_entity_id2k_dict(args["data"], args["sample"], args["gen"], args)
    logger.debug("loaded id2k: {}".format(id2k_dict))

    count = 0

    #   for each cluster
    for cluster in clusters:
    #       calculate max k
        k_values = list(map(lambda entity_id: id2k_dict[entity_id], cluster))
        max_k = max(k_values)

        logger.debug("max k values: {} among {}".format(max_k, k_values))

        #       if max k > num entities of cluster
        if cluster.num_entities >= max_k * 2:
            count += 1

    return count

clusters_metric_dict = {
    ADM_METRIC: calculate_adm,
    DM_METRIC: calculate_dm,
    AM_METRIC: calculate_am,
    REMOVED_ENTITIES_METRIC: calculate_removed_entities,
    RAW_ENTITIES_METRIC: calculate_raw_entities,
    NUM_INVALID_ANONYMITY_CLUSTERS_METRIC: calculate_num_invalid_anonymity_clusters,
    NUM_BIG_CLUSTERS_METRIC: calculate_num_big_clusters,
    REMAINING_ADM_METRIC: calculate_remaining_adm,
    # OUT_DM_METRIC: calculate_odm,
    # IN_DM_METRIC: calculate_idm,
    # FAKE_ENTITIES_METRIC: calculate_num_fake_entities,
    # REAL_ENTITIES_METRIC: calculate_num_real_entities,
    # REAL_EDGES_METRIC: calculate_num_real_edges,
    # FAKE_EDGES_METRIC: calculate_num_fake_edges,
    # ANONYMIZED_ANONYMITY_METRIC: calculate_anonymity,
}

def get_all_metric_names():
    return list(clusters_metric_dict.keys())

def calculate_quality_metrics(metric_names, clusters, graph, args):
    quality = {}
    for metric_name in metric_names:
        fn = clusters_metric_dict[metric_name]
        quality_value = fn(clusters, graph, args)
        quality[metric_name] = quality_value

    return quality

