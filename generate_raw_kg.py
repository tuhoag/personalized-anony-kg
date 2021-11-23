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
    graph = dutils.load_graph_from_raw_data(args["data"], args["sample"], args)
    logger.info(graph)

    output_path = putils.get_raw_graph_path(args["data"], args["sample"])
    graph.export(output_path)
    logger.info("exported to {}".format(output_path))


if __name__ == "__main__":
    args = rutils.setup_arguments(add_arguments)
    rutils.setup_console_logging(args)
    main(args)