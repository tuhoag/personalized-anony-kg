import argparse
import logging

import anonygraph.utils.runner as rutils
import anonygraph.utils.data as dutils
import anonygraph.utils.path as putils

logger = logging.getLogger(__file__)

def add_arguments(parser):
    rutils.add_data_argument(parser)
    rutils.add_log_argument(parser)


def main(args):
    logger.info(args)
    data_name = args["data"]
    sample = args["sample"]

    graph = dutils.load_raw_graph(data_name, sample)
    logger.info(graph)



if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)