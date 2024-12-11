from mmcv.transforms import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler

custom_imports = dict(
    imports=['mmdet_wifi.dataset','mmdet_wifi.loading','mmdet_wifi.resize'],
    allow_failed_imports=False)

from dataset import CSIDataset 
from loading import CSILoadImageFromFile
from resize import CSIResize
from formatting import CSIPackDetInputs




# from mmdet_wifi.configs.models import model
from mmdet.datasets import AspectRatioBatchSampler, CocoDataset
from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomFlip, Resize)
from mmdet.evaluation import CocoMetric

# dataset settings
dataset_type = CSIDataset
data_root = '/home/multisig/datasets/wimans/'


backend_args = None

train_pipeline = [
    dict(type=CSILoadImageFromFile, crop_length=32, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    # dict(type=Resize, scale=(960, 540), keep_ratio=True),
    dict(type=CSIResize, scale=(960, 540), keep_ratio=True),
    #dict(type=RandomFlip, prob=0.5),
    dict(type=CSIPackDetInputs,
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type=CSILoadImageFromFile,crop_length=32, backend_args=backend_args),
    # dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=CSIResize, scale=(960, 540), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=CSIPackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root='/home/multisig/datasets/wimans',
        # ann_file='annotations/instances_train2017.json',
        ann_file='annotation_coco_10x.json',
        data_prefix=dict(img='wifi_csi/amp'), #is this correct?
        filter_cfg=dict(environment='classroom', wifi_band=2.4),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type, #TODO: Check!!
        # data_root=data_root,
        # ann_file='annotations/instances_val2017.json',
        # data_prefix=dict(img='val2017/'),
        data_root='/home/multisig/datasets/wimans',
        # ann_file='annotations/instances_train2017.json',
        ann_file='annotation_coco_10x.json',
        data_prefix=dict(img='wifi_csi/amp'), 
        filter_cfg=dict(environment='classroom', wifi_band=2.4),
        test_mode=False,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file='/home/multisig/datasets/wimans/annotation_coco_10x.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
