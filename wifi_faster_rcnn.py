from typing import Tuple
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.two_stage import TwoStageDetector
from torch import Tensor

@MODELS.register_module()
class WifiFasterRCNN(TwoStageDetector):
    

    def __init__(self,
                 mtn: ConfigType,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.mtn = MODELS.build(mtn)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        batch_inputs = self.mtn(batch_inputs)
        return super().extract_feat(batch_inputs)


