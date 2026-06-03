_base_ = ["wedetect_base.py"]


load_from = 'wedetect_base_uni.pth'

num_training_classes = 256 # we mistakenly used 256 prompts, but actually only the first one is used.
num_classes = 256
text_channels = 768
neck_embed_channels = [128, 256, 512]
neck_num_heads = [4, 8, 16]

base_lr = 2e-5
weight_decay = 0.05
train_batch_size_per_gpu = 4
max_epochs = 4  # Maximum training epochs
close_mosaic_epochs = 2
save_epoch_intervals = 1
# persistent_workers must be False if num_workers is 0
persistent_workers = True

img_scale = (640, 640)
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
mixup_prob = 0.15

# model settings
model = dict(
    type='SimpleYOLOWorldDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    prompt_dim=text_channels,
    num_prompts=num_training_classes,
    freeze_prompt=False,
    use_mlp_adapter=False,
    data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        text_model=None,
        image_model=dict(
            type="ConvNextVisionBackbone",
            model_name='base',
            frozen_modules=['all'],
        ),
        with_text_model=False),
    neck=dict(
        type="CSPRepBiFPANNeck",
        freeze_all=False,
    ),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    freeze_all=False,
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))


# dataset settings

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]


mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='WeDetectRandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - affine_scale,
                             1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform,
                       *mosaic_affine_transform]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='WeDetectHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(
        type='WeDetectLetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='WeDetectRandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='WeDetectHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(
        type='WeDetectLetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
        ),
    ),
]


coco_train_dataset = dict(type='YOLOv5CocoDataset',
                          metainfo=dict(classes=['object']), # class-agnostic setting
                          data_root='data/coco/',
                          ann_file='data/coco/annotations/coco_instances_objectness.json',
                          data_prefix=dict(img='train2017/'),
                          filter_cfg=dict(filter_empty_gt=False, min_size=32),
                          pipeline=train_pipeline)


train_dataloader = dict(
    num_workers=2,
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=coco_train_dataset)


coco_val_dataset = dict(
    _delete_=True,
    type='YOLOv5CocoDataset',
    data_root='data/coco/',
    ann_file='data/coco/annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=test_pipeline
)


val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(metric='proposal_fast')
test_evaluator = val_evaluator

# training settings
param_scheduler = [
    # 1. 线性预热 (Warmup)
    dict(
        type='LinearLR',
        start_factor=0.001, # 预热开始时的倍率
        end_factor=1.0,     # 预热结束时的倍率
        begin=0,
        end=1000,           # 预热持续 1000 次迭代
        by_epoch=False      # 按 iteration 计数
    ),
    # 2. 主训练过程线性递减 (Main Linear Decay)
    dict(
        type='LinearLR',
        start_factor=1.0,   # 衰减开始倍率
        end_factor=0.001,   # 衰减结束倍率
        begin=0,            # 从第 0 epoch 开始计算衰减曲线
        end=max_epochs,     # 到第 12 epoch 结束
        by_epoch=True,      # 按 epoch 规划
        convert_to_iter_based=True # 转换为 iter 模式，实现平滑下降
    )
]

custom_hooks = [
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)])

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')



