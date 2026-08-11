_base_ = ["./wedetect_tiny.py"]

# ===========================================================================
# WeDetect-Tiny  <-- distill from -->  WeDetect-Large
# ===========================================================================
# Feature-level KD on the neck outputs (P3/P4/P5).
#   - Algorithm: FGD focal distillation (CVPR 2022) by default.
#                Set distill_loss.mode='cwd' to switch to CWD (ICCV 2023).
#   - Teacher: WeDetect-Large (frozen). Student: WeDetect-Tiny (trainable).
#   - Channel alignment: 1x1 conv per level (student [96,192,384] -> teacher
#     [192,384,768]).
#
# Why feature-level (not logit) KD:
#   WeDetect is retrieval-based: cls_logit = contrastive(region_embed,
#   text_embed). The teacher (xlm-roberta-large) and student (xlm-roberta-base)
#   have DIFFERENT text-embedding spaces, so their cls_logit spaces differ and
#   direct logit KD is invalid. Neck-output feature KD sidesteps this.
#
# Run:
#   bash dist_train.sh config/wedetect_tiny_distill_large.py 8 --amp
# Eval (COCO):
#   bash dist_test.sh config/wedetect_tiny_distill_large.py \
#       work_dirs/wedetect_tiny_distill_large/epoch_12.pth 8
# Eval (LVIS minival): comment val_evaluator below to use lvis_minival_evaluator.
# ===========================================================================

# --- checkpoints ------------------------------------------------------------
# Student init weights (WeDetect-Tiny) and teacher weights (WeDetect-Large).
load_from = 'checkpoints/wedetect_tiny.pth'
teacher_checkpoint = 'checkpoints/wedetect_large.pth'

# --- classes ----------------------------------------------------------------
# COCO finetuning variant (runnable out of the box).
num_training_classes = 80
num_classes = 80

base_lr = 2e-5
weight_decay = 0.05
train_batch_size_per_gpu = 8
max_epochs = 12
close_mosaic_epochs = 4
save_epoch_intervals = 1
persistent_workers = True
find_unused_parameters = True

img_scale = (640, 640)
affine_scale = 0.5
mixup_prob = 0.15

# --- student neck/head channels (for the alignment adapters) ----------------
student_channels = [96, 192, 384]   # tiny neck outputs (P3, P4, P5)
teacher_channels = [192, 384, 768]  # large neck outputs (P3, P4, P5)

# --- teacher model config (a full WeDetect-Large) ---------------------------
teacher_model = dict(
    type='YOLOWorldDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type='YOLOWDetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='ConvNextVisionBackbone',
            model_name='large',
            frozen_modules=[]),
        text_model=dict(
            type='XLMRobertaLanguageBackbone',
            model_name='./xlm-roberta-large/',
            model_size='large',
            frozen_modules=[])),
    neck=dict(
        type='CSPRepBiFPANNeck',
        scale_factor=1.5,
        model_size='large'),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=768,
            num_classes=num_training_classes,
            model_size='large',
            in_channels=[256, 512, 1024]),
        prior_generator=dict(
            type='MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='WeDetectDistancePointBBoxCoder'),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='mmyoloIoULoss', iou_mode='ciou', bbox_format='xyxy',
            reduction='sum', loss_weight=7.5, return_iou=False),
        loss_dfl=dict(
            type='DistributionFocalLoss', reduction='mean',
            loss_weight=1.5 / 4)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner', num_classes=num_classes,
            use_ciou=True, topk=10, alpha=0.5, beta=6.0, eps=1e-9)),
    test_cfg=dict(
        multi_label=True, nms_pre=30000, score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7), max_per_img=300))

# --- distillation loss ------------------------------------------------------
# mode='fgd'  : FGD focal distillation (GT-guided fg/bg + teacher channel
#               attention). Best for detection, needs GT boxes.
# mode='cwd'  : channel-wise KL divergence. No GT needed, very robust.
distill_loss = dict(
    type='NeckDistillLoss',
    mode='fgd',
    student_channels=student_channels,
    teacher_channels=teacher_channels,
    featmap_strides=[8, 16, 32],
    fg_weight=1.0,
    bg_weight=0.5,
    temperature=4.0,
    loss_weight=1.0)

# --- override the student model to be a distillation detector ----------------
# NOTE: the student backbone/neck/bbox_head below are the *tiny* ones inherited
# from wedetect_tiny.py; only the wrapping detector type and KD args change.
# ``_delete_=True`` fully replaces the base ``model`` to avoid merge surprises.
model = dict(
    _delete_=True,
    type='DistillYOLOWorldDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type='YOLOWDetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='ConvNextVisionBackbone',
            model_name='tiny',
            frozen_modules=[]),
        text_model=dict(
            type='XLMRobertaLanguageBackbone',
            model_name='./xlm-roberta-base/',
            model_size='tiny',
            frozen_modules=[])),
    neck=dict(type='CSPRepBiFPANNeck', model_size='tiny'),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=768,
            num_classes=num_training_classes,
            model_size='tiny',
            in_channels=[256, 512, 1024]),
        prior_generator=dict(
            type='MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='WeDetectDistancePointBBoxCoder'),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='mmyoloIoULoss', iou_mode='ciou', bbox_format='xyxy',
            reduction='sum', loss_weight=7.5, return_iou=False),
        loss_dfl=dict(
            type='DistributionFocalLoss', reduction='mean',
            loss_weight=1.5 / 4)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner', num_classes=num_classes,
            use_ciou=True, topk=10, alpha=0.5, beta=6.0, eps=1e-9)),
    test_cfg=dict(
        multi_label=True, nms_pre=30000, score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7), max_per_img=300),
    # ---- KD-specific args ----
    teacher_cfg=teacher_model,
    teacher_checkpoint=teacher_checkpoint,
    distill_loss=distill_loss,
    student_channels=student_channels,
    teacher_channels=teacher_channels,
    save_teacher=False)

# --- train pipeline (same as the COCO full-tuning config) -------------------
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]

text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
mosaic_affine_transform = [
    dict(type='MultiModalMosaic', img_scale=img_scale, pad_val=114.0,
         pre_transform=pre_transform),
    dict(type='WeDetectRandomAffine',
         max_rotate_degree=0.0, max_shear_degree=0.0, max_aspect_ratio=100.,
         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
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
    dict(type='YOLOv5MultiModalMixUp', prob=mixup_prob,
         pre_transform=[*pre_transform, *mosaic_affine_transform]),
    dict(type='mmdet.Albu', transforms=albu_train_transforms,
         bbox_params=dict(type='BboxParams', format='pascal_voc',
                          label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
         keymap={'img': 'image', 'gt_bboxes': 'bboxes'}),
    dict(type='WeDetectHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    *text_transform
]
train_pipeline_stage2 = [
    *pre_transform,
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(type='WeDetectLetterResize', scale=img_scale, allow_scale_up=True,
         pad_val=dict(img=114.0)),
    dict(type='WeDetectRandomAffine',
         max_rotate_degree=0.0, max_shear_degree=0.0,
         scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
         max_aspect_ratio=100, border_val=(114, 114, 114)),
    dict(type='mmdet.Albu', transforms=albu_train_transforms,
         bbox_params=dict(type='BboxParams', format='pascal_voc',
                          label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
         keymap={'img': 'image', 'gt_bboxes': 'bboxes'}),
    dict(type='WeDetectHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    *text_transform
]

# --- dataset ----------------------------------------------------------------
# Default: COCO finetuning (runnable out of the box).
# For multi-dataset PRETRAINING (goldg + v3det + obj365 + openimagesv6),
# replace the single dataset below with a WeConcatDataset of MultiModalDataset
# wrappers (one per source) and set num_classes/num_training_classes to the
# union vocabulary size. See the README distillation section for a template.
coco_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='WeCocoDataset',
        data_root='data/coco/',
        ann_file='data/coco/annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_zh_class_texts.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    num_workers=2,
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=coco_train_dataset)

# --- training settings ------------------------------------------------------
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, end_factor=1.0,
         begin=0, end=1000, by_epoch=False),
    dict(type='LinearLR', start_factor=1.0, end_factor=0.001,
         begin=0, end=max_epochs, by_epoch=True, convert_to_iter_based=True)
]

custom_hooks = [
    dict(type='mmdet.PipelineSwitchHook',
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
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0),
            # distillation alignment adapters: slightly smaller lr
            'distill_loss': dict(lr_mult=1.0),
        }),
    constructor='YOLOWv5OptimizerConstructor')

# --- evaluation -------------------------------------------------------------
# COCO by default. To evaluate on LVIS minival, swap the two lines below.
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox')
# val_evaluator = dict(
#     type='LVISMetric',
#     ann_file='data/lvis/lvis_v1_minival_inserted_image_name.json',
#     metric='bbox')
test_evaluator = val_evaluator
