import numpy as np
from tqdm import tqdm
from glob import glob
import argparse
import logging
import os
import itertools
from joblib import Parallel, delayed

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils
import anonygraph.utils.general as utils
import anonygraph.k_generators as generators

logger = logging.getLogger(__file__)


def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_k_generator_runner_argument(parser)
    rutils.add_workers_argument(parser)
    rutils.add_log_argument(parser)


def main(args):
    logger.info(args)
    n_gens = args["n_gens"]
    gen_name = args["gen"]

    # get real num of gens
    real_n_gens = generators.get_real_num_generations(gen_name, n_gens)
    gen_args_list = args["gen_args_list"]

    args_list = []
    for gen_n, gen_args in itertools.product(range(real_n_gens), gen_args_list):
        current_args = args.copy()
        current_args["gen_n"] = gen_n
        current_args["gen_args"] = gen_args

        args_list.append(current_args)

    file_name = 'generate_k_values.py'
    Parallel(n_jobs=args['workers'])(delayed(rutils.run_python_file)(file_name, args_item) for args_item in args_list)


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)
