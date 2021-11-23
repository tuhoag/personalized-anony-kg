import itertools
import argparse
import logging
from joblib import Parallel, delayed

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.algorithms.clustering as calgo
import anonygraph.k_generators as generators

logger = logging.getLogger(__file__)

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_clustering_runner_argument(parser)
    rutils.add_k_generator_runner_argument(parser)
    rutils.add_info_loss_argument(parser)
    rutils.add_log_argument(parser)
    rutils.add_workers_argument(parser)


def main(args):
    logger.info(args)
    n_gens = args["n_gens"]
    gen_name = args["gen"]
    max_cost_list = args["max_cost_list"]

    # get real num of gens
    real_n_gens = generators.get_real_num_generations(gen_name, n_gens)

    args_list = []
    for gen_n, max_cost in itertools.product(range(real_n_gens), max_cost_list):
        current_args = args.copy()
        current_args["gen_n"] = gen_n
        current_args["max_cost"] = max_cost

        args_list.append(current_args)

    file_name = 'generate_raw_clusters.py'
    Parallel(n_jobs=args['workers'])(delayed(rutils.run_python_file)(file_name, args_item) for args_item in args_list)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)