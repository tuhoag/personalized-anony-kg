import logging

from .adm import AttributeOutInDegreeInfoLoss
from .am import AttributeInfoLoss
from .dm import OutDegreeInfoLoss, InDegreeInfoLoss, OutInDegreeInfoLoss
from anonygraph.constants import *


logger = logging.getLogger(__name__)

def get_info_loss_function(info_loss_name, graph, args):
    info_loss_args = args["info_loss_args"]
    # raise Exception("args: {}".format(args))

    if info_loss_name == AM_METRIC:
        info_loss_fn = AttributeInfoLoss(graph, info_loss_args)
    elif info_loss_name == ADM_METRIC:
        info_loss_fn = AttributeOutInDegreeInfoLoss(graph, info_loss_args)
    elif info_loss_name == DM_METRIC:
        info_loss_fn = OutInDegreeInfoLoss(graph, info_loss_args)
    else:
        raise NotImplementedError(
            "Unsupported info loss: {}".format(info_loss_name)
        )

    return info_loss_fn