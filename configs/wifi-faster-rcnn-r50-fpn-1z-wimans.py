from mmengine.config import read_base

with read_base():
    from ..mmdetection.mmdet.configs._base_.default_runtime import *
    from ..mmdetection.mmdet.configs._base_.schedules.schedule_1x import *
    from ..configs.dataset import *
    from ..configs.models import *

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.002, momentum=0.9, weight_decay=0.0001))
