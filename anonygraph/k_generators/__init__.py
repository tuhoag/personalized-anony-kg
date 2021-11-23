import logging

from .total_degree_generator import TotalDegreeGenerator
from .range_total_degree_generator import RangeTotalDegreeGenerator
from .same_value_generator import SameValueGenerator
from .random_normal_generator import RandomNormalGenerator
from .ratio_random_normal_generator import RatioRandomNormalGenerator
from .ratio_zipf_generator import RatioZipfGenerator
from .range_zipf_generator import RangeZipfGenerator
from .static_generator import StaticGenerator

from anonygraph.constants import *

logger = logging.getLogger(__name__)

def get_generator(gen_name, args):
    splits = args["gen_args"]
    if gen_name == SAME_K_GENERATOR:
        k = int(splits[0])
        gen_fn = SameValueGenerator(k)
    elif gen_name == STATIC_GENERATOR:
        gen_fn = StaticGenerator(args["data"])
    elif gen_name == NORM_LIST_K_GENERATOR:
        min_k_ratio = int(splits[0])
        max_k_ratio = int(splits[1])
        step_k_ratio = int(splits[2])
        gen_fn = RandomNormalGenerator(range(min_k_ratio, max_k_ratio, step_k_ratio))
    elif gen_name == RATIO_NORM_GENERATOR:
        max_k_ratio = float(splits[0])
        num_k_vals = int(splits[1])
        gen_fn = RatioRandomNormalGenerator(max_k_ratio, num_k_vals)
    elif gen_name == RATIO_ZIPF_GENERATOR:
        logger.debug(splits)
        min_k_ratio = float(splits[0])
        max_k_ratio = float(splits[1])
        step_k_ratio = float(splits[2])
        param = float(splits[3])
        gen_fn = RatioZipfGenerator(min_k_ratio, max_k_ratio, step_k_ratio, param)
    elif gen_name == RANGE_ZIPF_GENERATOR:
        min_k = int(splits[0])
        max_k = int(splits[1])
        step_k = int(splits[2])
        param = float(splits[3])
        gen_fn = RangeZipfGenerator(min_k, max_k, step_k, param)
    elif gen_name == TOTAL_DEGREE_GENERATOR:
        min_k_ratio = float(splits[0])
        max_k_ratio = float(splits[1])
        step_k_ratio = float(splits[2])
        gen_fn = TotalDegreeGenerator(min_k_ratio, max_k_ratio, step_k_ratio)
    elif gen_name == RANGE_TOTAL_EDGES_GENERATOR:
        min_k = int(splits[0])
        max_k = int(splits[1])
        step_k = int(splits[2])
        gen_fn = RangeTotalDegreeGenerator(min_k, max_k, step_k)
    else:
        raise NotImplementedError("Unsupported k gen: {}".format(gen_name))

    return gen_fn

def get_real_num_generations(gen_name, num_gens):
    if gen_name in [SAME_K_GENERATOR, TOTAL_DEGREE_GENERATOR, RANGE_TOTAL_EDGES_GENERATOR]:
        return 1
    elif gen_name in [RATIO_NORM_GENERATOR, NORM_LIST_K_GENERATOR, RATIO_ZIPF_GENERATOR, RANGE_ZIPF_GENERATOR]:
        return num_gens
    else:
        raise NotImplementedError("Unsupported generator: {}".format(gen_name))