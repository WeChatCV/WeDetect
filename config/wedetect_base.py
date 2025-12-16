_base_ = ["default_runtime.py"]

# hyper-parameters
num_classes = 1203
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 2
save_epoch_intervals = 1
text_channels = 768
neck_embed_channels = [128, 256, 512]
neck_num_heads = [4, 8,16]
base_lr = 5e-4
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 10

find_unused_parameters = True

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4

custom_imports = dict(imports=["wedetect"], allow_failed_imports=False)
# model settings
model = dict(
    type="YOLOWorldDetector",
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        type="YOLOWDetDataPreprocessor",
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type="MultiModalYOLOBackbone",
        image_model=dict(
            type="ConvNextVisionBackbone",
            model_name="base",
            frozen_modules=[],
        ),
        text_model=dict(
            type="XLMRobertaLanguageBackbone",
            model_name="./xlm-roberta-base/",
            model_size="base",
            frozen_modules=[],
        ),
    ),
    neck=dict(
        type="CSPRepBiFPANNeck",
        scale_factor=1.0,
        model_size = 'base',
    ),
    bbox_head=dict(
        type="YOLOWorldHead",
        head_module=dict(
            type="YOLOWorldHeadModule",
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes,
            model_size = 'base',
            in_channels=[256, 512, 1024],
        ),
        prior_generator=dict(
            type='MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='WeDetectDistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='mmyoloIoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    test_cfg=model_test_cfg)

img_scale = (640, 640)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(
        type='WeDetectLetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type="LoadText"),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
            "texts",
        ),
    ),
]

dist_cfg = dict(backend="nccl", timeout=10800)


coco_val_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="WeCocoDataset",
        data_root="data/coco/",
        test_mode=True,
        ann_file="data/coco/annotations/instances_val2017.json",
        data_prefix=dict(img="val2017"),
        batch_shapes_cfg=None,
    ),
    class_text_path="data/texts/coco_zh_class_texts.json",
    pipeline=test_pipeline,
)


lvis_minival_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="YOLOv5LVISV1Dataset",
        data_root=f"data/coco/",
        test_mode=True,
        ann_file=f"data/lvis/lvis_v1_minival_inserted_image_name.json",
        data_prefix=dict(img=""),
        batch_shapes_cfg=None,
    ),
    class_text_path=f"data/texts/lvis_v1_zh_class_texts.json",
    pipeline=test_pipeline,
)

lvis_od_val_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="YOLOv5LVISV1Dataset",
        data_root=f"data/coco/",
        test_mode=True,
        ann_file="data/lvis/lvis_od_val.json",
        data_prefix=dict(img=""),
        batch_shapes_cfg=None,
    ),
    class_text_path=f"data/texts/lvis_v1_zh_class_texts.json",
    pipeline=test_pipeline,
)
coco_evaluator = dict(
    type="CocoMetric",
    ann_file=f"data/coco/annotations/instances_val2017.json",
    metric="bbox",
)

lvis_minival_evaluator = dict(
    type="LVISMetric",
    ann_file=f"data/lvis/lvis_v1_minival_inserted_image_name.json",
    metric="bbox",
)
lvis_od_val_evaluator = dict(
    type="LVISMetric",
    ann_file="data/lvis/lvis_od_val.json",
    metric="bbox",
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=coco_val_dataset)

test_dataloader = val_dataloader

# val_evaluator = _base_.lvis_minival_evaluator
# val_evaluator = _base_.lvis_od_val_evaluator
val_evaluator = coco_evaluator
test_evaluator = val_evaluator


val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

