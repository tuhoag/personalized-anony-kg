import os
from anonygraph.constants import ASSERTION_VARIABLE, NO_REMOVAL_HANDLER
import subprocess
import logging
import argparse
import numpy as np

import anonygraph.utils.path as putils

logger = logging.getLogger(__name__)

class str2range():
    def __init__(self, out_type):
        self.type = out_type

    def __call__(self, value):
        start, stop, interval = map(self.type, value.split(","))


        return range(start, stop, interval)

class str2list():
    def __init__(self, out_type, delimiter=","):
        self.type = out_type
        self.delimiter = delimiter

    def __call__(self, value):
        values = []

        if value is not None:
            values = list(map(lambda str: self.type(str), value.split(self.delimiter)))

        return values


def str2bool(value):
    if value is None:
        return None

    if value in ["yes", "True", "y"]:
        return True
    elif value in ["no", "False", "n"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def str2log_mode(value):
    if value is None:
        return None

    if value in ["d", "debug", "10"]:
        log_mode = logging.DEBUG
    elif value in ["i", "info", "20"]:
        log_mode = logging.INFO
    elif value in ["w", "warning", "30"]:
        log_mode = logging.WARNING
    else:
        raise argparse.ArgumentTypeError("Unsupported log mode type: {}".format(value))

    return log_mode


def setup_arguments(add_arguments_fn):
    parser = argparse.ArgumentParser(description="Process some integers.")

    add_arguments_fn(parser)

    args, _ = parser.parse_known_args()

    params = {}
    for arg in vars(args):
        params[arg] = getattr(args, arg)

    os.environ[ASSERTION_VARIABLE] = params["assert"]

    return params

def setup_console_logging(args):
    level = args["log"]

    logger = logging.getLogger("")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(name)-12s[%(lineno)d]: %(funcName)s %(levelname)-8s %(message)s "
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)


def add_data_argument(parser):
    parser.add_argument("--data")
    parser.add_argument("--sample", type=int, default=-1)

def add_log_argument(parser):
    parser.add_argument("--log", type=str2log_mode, default=logging.INFO)
    parser.add_argument("--assert", default="no")



def convert_raw_val_to_str_val(name, val):
    new_val = None
    if val is None:
        val = None
    elif type(val) is list:
        new_val = []
        for current_val in val:
            new_val.append(convert_raw_val_to_str_val(name, current_val))
        new_val = putils.get_str_delimiter(name).join(map(str, new_val))
    elif type(val) is bool:
        if val:
            new_val = "y"
        else:
            new_val = "n"
    elif type(val) is float:
        new_val = "{:.2f}".format(val)
        # logger.debug("new: {} - old: {}".format(new_val, val))
        # raise Exception()
    else:
        new_val = str(val)

    return new_val


def copy_args(current_args):
    new_args = {}

    for name, val in current_args.items():
        logger.debug("name: {} - val: {} - type: {} - is list: {}".format(name, val, type(val), type(val) is list))

        new_args[name] = convert_raw_val_to_str_val(name, val)

        # if name == "run_mode":
        #     raise Exception()
    return new_args

def add_args_list_argument(parser, name, type=str, default=None):
    parser.add_argument("--{}".format(name), type=str2list(type, delimiter=putils.get_str_delimiter(name)), default=default)

def add_info_loss_argument(parser):
    parser.add_argument("--info_loss", default="adm")
    add_args_list_argument(parser, "info_loss_args", float, "0.5,0.5")


def add_workers_argument(parser):
    parser.add_argument("--workers", type=int)

def add_k_generator_param_argument(parser):
    parser.add_argument("--gen")
    add_args_list_argument(parser, "gen_args")

def add_k_generator_argument(parser):
    add_k_generator_param_argument(parser)
    parser.add_argument("--gen_n", type=int, default=0)

def add_k_generator_runner_argument(parser):
    add_k_generator_param_argument(parser)
    parser.add_argument("--n_gens", type=int, default=1)
    add_args_list_argument(parser, "gen_args_list")

def add_clustering_argument(parser):
    parser.add_argument("--calgo")
    add_args_list_argument(parser, "calgo_args")
    parser.add_argument("--handler", default=NO_REMOVAL_HANDLER)
    add_args_list_argument(parser, "handler_args", type=float)
    # parser.add_argument("--handler_args", type=float)

def add_clustering_runner_argument(parser):
    parser.add_argument("--calgo")
    add_args_list_argument(parser, "calgo_args_list")
    parser.add_argument("--handler")
    add_args_list_argument(parser, "handler_args", type=float)
    # add_args_list_argument(parser, "handler_args_list", float, "0")
    # parser.add_argument("--handler_args_list", type=str2list(float), default=[0])

def add_cluster_constraint_enforcer_argument(parser):
    parser.add_argument("--enforcer")
    add_args_list_argument(parser, "enforcer_args", float)

def add_cluster_constraint_enforcer_runner_argument(parser):
    parser.add_argument("--enforcer")
    add_args_list_argument(parser, "enforcer_args_list")

def add_points_argument(parser):
    """
    Add arguments to generate points.
    """
    parser.add_argument("--d", type=int)

def add_learner_argument(parser):
    """
    Add arguments to learners.
    """
    parser.add_argument("--gpu")
    parser.add_argument("--batch",type=int)
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--continue", type=str2bool)

def run_python_file(path, args):
    arguments = ["python", path]
    for name, value in args.items():
        if value is not None:
            arguments.append("--" + name)
            arguments.append(str(value))

    logger.debug("run {}: {}".format(path, arguments))

    with subprocess.Popen(arguments, stdout=subprocess.PIPE) as process:
        for line in iter(process.stdout):
            logger.debug(line.rstrip().decode("utf-8"))

        process.communicate()
        return process.returncode