import logging

logger = logging.getLogger(__name__)

from .base_info_loss import BaseInfoLossMetric
from .am import AttributeInfoLoss
from .dm import OutDegreeInfoLoss, InDegreeInfoLoss, OutInDegreeInfoLoss


class AttributeOutInDegreeInfoLoss(BaseInfoLossMetric):
    def __init__(self, graph, args):
        super().__init__(graph, args)

        self.attribute_info_loss_fn = AttributeInfoLoss(graph, None)

        # raise Exception("args: {}".format(args))
        self.out_in_degree_info_loss_fn = OutInDegreeInfoLoss(graph, [args[1]])

        self.alpha_adm = args[0]

    def call(self, entity_ids):
        attribute_info_loss = self.attribute_info_loss_fn.call(entity_ids)
        degree_info_loss = self.out_in_degree_info_loss_fn.call(entity_ids)

        info_loss = attribute_info_loss * self.alpha_adm + (1 - self.alpha_adm) * degree_info_loss

        return info_loss
