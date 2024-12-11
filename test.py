from mmengine.config import Config
from mmdet.registry import MODELS, DATASETS
import torch

import json

with open("/home/multisig/datasets/wimans/annotation_coco.json") as f:
    data = json.load(f)

#cfg = Config.fromfile("/home/multisig/repos/mmdet_wifi/configs/models.py")
#cfg_dataset = Config.fromfile("/home/multisig/repos/mmdet_wifi/configs/dataset.py")
#model = MODELS.build(cfg.model)
#dataset = DATASETS.build(cfg_dataset)

#print(cfg_dataset)

#x = torch.rand((1, 30, 3, 3, 30))
#x = model.extract_feat(x)
#for item in x:
#    print(item.shape)
