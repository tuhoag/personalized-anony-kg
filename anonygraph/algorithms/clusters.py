import os
import logging

logger = logging.getLogger(__name__)

import anonygraph.algorithms as algo

class Clusters(object):
    def __init__(self):
        self.__clusters = []

    def add_cluster(self, cluster):
        self.__clusters.append(cluster)

    def add_clusters(self, clusters):
        self.__clusters.extend(clusters)

    def to_file(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'w+') as f:
            for cluster in self.__clusters:
                line = "{}\n".format(cluster.to_line_str())

                f.write(line)

    def __iter__(self):
        for cluster in self.__clusters:
            yield cluster

    @staticmethod
    def from_clusters_iter(clusters_iters):
        clusters = algo.Clusters()

        for cluster_set in clusters_iters:
            clusters.add_cluster(algo.Cluster.from_iter(cluster_set))

        return clusters

    @staticmethod
    def from_file(path):
        clusters = algo.Clusters()

        with open(path, 'r') as file:
            for line in file:
                entity_ids = list(map(int, line.strip().split(',')))
                cluster = algo.Cluster.from_iter(entity_ids)
                clusters.add_cluster(cluster)

        return clusters

    def __getitem__(self, index):
        return self.__clusters[index]

    def __str__(self):
        return "".join(map(str, self.__clusters))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.__clusters)

    def pop(self, index=0):
        return self.__clusters.pop(index)

    def insert(self, index, cluster):
        self.__clusters.insert(index, cluster)

    @property
    def entity_ids(self):
        entity_ids = []
        for cluster in self.__clusters:
            entity_ids.extend(cluster.to_list())

        return entity_ids

    def has_entity_id(self, entity_id):
        for cluster in self.__clusters:
            if cluster.has_entity_id(entity_id):
                return True

        return False

    def find_invalid_clusters(self, graph, checker_fn):
        invalid_clusters = []

        for cluster in self:
            # TODO get user in - out bound
            if not checker_fn.is_cluster_valid(cluster, graph):
            # if not cluster.is_valid(self.graph):
                invalid_clusters.append(cluster)

        return invalid_clusters


