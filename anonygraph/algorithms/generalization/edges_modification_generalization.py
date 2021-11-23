import logging
import copy

from .same_attributes_gen import SameAttributesGenerator
from .same_degree_relationships_gen import SameDegreeRelationshipsGenerator
from .unused_entities_removal_gen import UnusedEntitiesRemovalGenerator

logger = logging.getLogger(__name__)

class EdgesModificationGeneralization(object):
    def __init__(self, graph):
        self.graph = graph

    def __call__(self, clusters):
        logger.debug("clusters: {}".format(clusters))
        new_graph = copy.deepcopy(self.graph)

        modification_algos = [
        ]

        if len(clusters) > 0:
            modification_algos.extend([
                UnusedEntitiesRemovalGenerator(),
                SameAttributesGenerator(),
                SameDegreeRelationshipsGenerator(),
            ])

        for algo_index, algo_fn in enumerate(modification_algos):
            algo_fn(new_graph, clusters)
            logger.debug("step: {} - graph: {}".format(algo_index, new_graph))

        return new_graph