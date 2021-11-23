from .small_removal_enforcer import SmallRemovalEnforcer
from .merge_split_enforcer import MergeSplitEnforcer
from .same_size_merge_split_enforcer import SameSizeMergeSplitEnforcer
from anonygraph.constants import *

def get_enforcer(enforcer_name, args):
    enforcer_args = args["enforcer_args"]

    if enforcer_name == SR_ENFORCER:
        return SmallRemovalEnforcer(enforcer_args)
    elif enforcer_name == MERGE_SPLIT_ENFORCER:
        return MergeSplitEnforcer(enforcer_args)
    elif enforcer_name == KMEANS_PARTITION_ENFORCER:
        return SameSizeMergeSplitEnforcer(enforcer_args)
    else:
        raise Exception("Do not support enforcer: {}".format(enforcer_name))