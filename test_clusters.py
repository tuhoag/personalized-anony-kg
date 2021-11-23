import glob
import logging
import anonygraph.utils.visualization as vutils
import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.algorithms.clustering as calgo
import anonygraph.algorithms as algo

logger = logging.getLogger(__name__)

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)

    parser.add_argument("--refresh", type=rutils.str2bool)


def test_clusters(path):
    #   load clusters
    clusters = algo.Clusters.from_file(path)

    #   extract info from path
    info = putils.extract_info_from_clusters_path(path)
    logger.debug("extracted {} from {}".format(info, path))

    #   load k values of that clusters
    id2k_dict = dutils.load_entity_id2k_dict(info["data"], info["sample"], info["gen_name"], info)
    logger.debug("loaded id2k: {}".format(id2k_dict))

    info["is_valid_anonymity"] = True

    #   for each cluster
    for cluster in clusters:
    #       calculate max k
        k_values = list(map(lambda entity_id: id2k_dict[entity_id], cluster))
        max_k = max(k_values)

        logger.debug("max k values: {} among {}".format(max_k, k_values))

        #       if max k > num entities of cluster
        if max_k > cluster.num_entities:
            info["is_valid_anonymity"] = False
            break

    return info

def test_valid_anonymity_clusters(data_name, sample, num_workers):
    # find all anonymized clusters paths
    dir_path = putils.get_clusters_dir_path(data_name, sample, "anony")
    logger.debug(dir_path)

    clusters_paths = glob.glob(dir_path + "/*.txt")
    logger.debug("found {} clusters {}".format(len(clusters_paths), clusters_paths))

    test_clusters(clusters_paths[0])

    pass

def main(args):
    logger.debug(args)
    data_name = args["data"]
    sample = args["sample"]
    num_workers = args["workers"]

    test_valid_anonymity_clusters(data_name, sample, num_workers)



if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
