import logging

logger = logging.getLogger(__name__)

def get_attribute_info(subgraph, entity_id):
    result = {}

    for entity_id, relation_id, value_id in subgraph.get_attribute_edges_iter_of_entity_id(entity_id):
        relation_value_ids = result.get(relation_id)

        if relation_value_ids is None:
            relation_value_ids = set()
            result[relation_id] = relation_value_ids

        relation_value_ids.add(value_id)

    return result

def get_generalized_attribute_info(subgraph, entity_ids):
    users_info = {}

    for entity_id in entity_ids:
        users_info[entity_id] = get_attribute_info(subgraph, entity_id)

    union_info = {}

    for entity_id, info in users_info.items():
        for relation_id, value_ids in info.items():
            union_value_ids = union_info.get(relation_id)

            if union_value_ids is None:
                union_value_ids = set()
                union_info[relation_id] = union_value_ids

            union_value_ids.update(value_ids)

    return union_info, users_info

def get_relationship_edges_fn(subgraph, entity_id, degree_type):
    if degree_type == "out":
        fn = lambda subgraph, entity_id: subgraph.get_out_relationship_edges_iter_of_entity_id(entity_id)
    elif degree_type == "in":
        fn = lambda subgraph, entity_id: subgraph.get_in_relationship_edges_iter_of_entity_id(entity_id)
    else:
        raise NotImplementedError("Unsupported degree type: {}".format(degree_type))

    return fn


def get_degree_info(subgraph, entity_id, degree_type):
    fn = get_relationship_edges_fn(subgraph, entity_id, degree_type)

    result = {}

    for _, relation_id, _ in fn(subgraph, entity_id):
        degree = result.get(relation_id)

        if degree is None:
            degree = 0

        degree += 1
        result[relation_id] = degree

    return result


def get_generalized_degree_info(subgraph, entity_ids, degree_type):
    entities_info = {}

    for entity_id in entity_ids:
        entities_info[entity_id] = get_degree_info(subgraph, entity_id, degree_type)

    union_info = {}

    for entity_id, info in entities_info.items():
        for relation_id, user_degree in info.items():
            union_degree = union_info.get(relation_id, 0)
            union_degree = max(union_degree, user_degree)
            union_info[relation_id] = union_degree

    return union_info, entities_info

def get_degree(subgraph, entity_id, relation_id, degree_type):
    fn = get_relationship_edges_fn(subgraph, entity_id, degree_type)

    result = 0

    for _, current_relation_id, _ in fn(subgraph, entity_id):
        if current_relation_id == relation_id:
            result +=1

    return result



def get_generalized_degree(subgraph, entity_ids, relation_id, degree_type):
    generalized_degree = 0

    for entity_id in entity_ids:
        entity_info = get_degree_info(subgraph, entity_id, degree_type)
        entity_degree = entity_info.get(relation_id, 0)

        generalized_degree = max(entity_degree, generalized_degree)

    return generalized_degree

def get_attribute_and_degree_info(subgraph, entity_id):
    info = {
        "attr": get_attribute_info(subgraph, entity_id),
        "out": get_degree_info(subgraph, entity_id, "out"),
        "in": get_degree_info(subgraph, entity_id, "in"),
    }

    return info

def get_out_in_degree_info(subgraph, entity_id):
    info = {
        "out": get_degree_info(subgraph, entity_id, "out"),
        "in": get_degree_info(subgraph, entity_id, "in"),
    }

    return info

def get_attribute_info_key(subgraph, entity_id):
    entity_info = get_attribute_info(subgraph, entity_id)
    relation_ids = sorted(entity_info.keys())

    result = []
    for relation_id in relation_ids:
        value_ids = tuple(sorted(entity_info[relation_id]))
        result.append((relation_id, value_ids))

    return tuple(result)

def get_degree_info_key(subgraph, entity_id, degree_type):
    entity_info = get_degree_info(subgraph, entity_id, degree_type)
    relation_ids = sorted(entity_info.keys())

    result = []
    for relation_id in relation_ids:
        degree = entity_info[relation_id]
        result.append((relation_id, degree))

    return tuple(result)

def get_out_in_degree_info_key(subgraph, entity_id):
    info_key = (
        get_degree_info_key(subgraph, entity_id, "out"),
        get_degree_info_key(subgraph, entity_id, "in")
    )

    return info_key

def get_attribute_and_degree_info_key(subgraph, entity_id):
    info_key = (
        get_attribute_info_key(subgraph, entity_id),
        get_degree_info_key(subgraph, entity_id, "out"),
        get_degree_info_key(subgraph, entity_id, "in")
    )

    return info_key