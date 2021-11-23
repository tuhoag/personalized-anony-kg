from anonygraph.constants import *
from .cost_zscore_handler import CostZscoreHandler
from .no_removal_handler import NoRemovalHandler

def get_outliers_handler(handler_name, args):
    if handler_name == COST_ZSCORE_HANDLER:
        max_cost = args["max_cost"]
        fn = CostZscoreHandler(max_cost)
    elif handler_name == NO_REMOVAL_HANDLER:
        fn = NoRemovalHandler()
    else:
        raise Exception("Unsupported outliers handlers: {}".format(handler_name))

    return fn