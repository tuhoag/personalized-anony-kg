import logging

from .edges_modification_generalization import EdgesModificationGeneralization

logger = logging.getLogger(__name__)
class GraphGeneralization(object):
    def __init__(self):
        pass

    def run(self, graph, clusters):
        logger.debug("clusters: {}".format(clusters))

        gen_fn = EdgesModificationGeneralization(graph)

        anony_graph = gen_fn(clusters)
        return anony_graph