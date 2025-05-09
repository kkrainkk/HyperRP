# 配置文件路径: configs/yolov8/yolov8_l_vg_grounding.py
_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world','yolo_world.datasets.V100'], allow_failed_imports=False)


# Hyper-Parameters
num_classes = 80  # 开放词汇总量
num_training_classes = 200  # 根据Flickr30k实际描述数调整
max_epochs = 1
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16
load_from = '/root/autodl-tmp/tools/work_dirs/Pretrain/epoch_2.pth'

# ------------------------ 模型架构 ------------------------
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='/root/autodl-tmp/CLIP_checkpoints',
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))


# ------------------------ 数据流水线 ------------------------
# Data Pipeline

# Data Pipeline
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
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]


train_pipeline = [
    *_base_.pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=_base_.mixup_prob,
        pre_transform=[*_base_.pre_transform,
                       *mosaic_affine_transform]),
    *_base_.last_transform[:-1],
    *text_transform
]
train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    *text_transform
]
# ------------------------ VG 数据集配置 ------------------------



# Flickr30k Dataset Configuration
vg_train_dataset1 = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='/root/autodl-tmp',
    ann_file='/root/autodl-tmp/vg_coco_format.json',
    data_prefix=dict(img='VG_100K/'),
    filter_cfg=dict(
        #filter_empty_gt=True,  # Flickr要求必须有标注
        min_size=32,
        max_gt_bbox=20),       # 限制每图最大标注数防止OOM
    pipeline=train_pipeline)


vg_train_dataset2 = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='/root/autodl-tmp',
    ann_file='/root/autodl-tmp/vg_coco_format.json',
    data_prefix=dict(img='VG_100K_2/'),
    filter_cfg=dict(
        #filter_empty_gt=True,  # Flickr要求必须有标注
        min_size=32,
        max_gt_bbox=20),       # 限制每图最大标注数防止OOM
    pipeline=train_pipeline)




train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    persistent_workers=False,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[vg_train_dataset1,vg_train_dataset2],  # 仅保留Flickr
        ignore_keys=['classes', 'palette']
    ),
)


# Validation Configuration (保持COCO验证)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]

coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/root/autodl-tmp/COCO/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017coco/'),
        test_mode=True,
        batch_shapes_cfg=None),
    class_text_path='/root/autodl-tmp/data/texts/coco_class_texts.json',
    pipeline=test_pipeline)

val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,  # 每5个epoch验证一次
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, _base_.val_interval_stage2)]
)

# ------------------------ 优化器配置 ------------------------
# Optimizer Configuration
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),  # 控制文本模型的学习率
            'logit_scale': dict(weight_decay=0.0),  # 防止 logit_scale 发生 L2 正则化
           # 'bbox_head': dict(lr_mult=0.0, decay_mult=0.0)  # 🔥 冻结 Head
        }
    ),
    constructor='YOLOWv5OptimizerConstructor',
)






# ------------------------ 评估与训练策略 ------------------------
# Evaluation Metrics
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='/root/autodl-tmp/COCO/annotations/instances_val2017.json',
    metric='bbox')


test_evaluator = dict(
    type='mmdet.CocoMetric',  # 使用 COCO 的评估指标
    proposal_nums=(100, 1, 10),
    ann_file='/root/autodl-tmp/COCO/annotations/instances_val2017.json',
    metric='bbox')

# Training Schedule
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        save_best='auto',  # 自动选择最佳模型
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50)  # 每50步记录日志
)

custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=0,
         switch_pipeline=train_pipeline_stage2)
]


