# 配置文件路径: configs/yolov8/yolov8_l_vg_grounding.py
_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world','yolo_world.datasets.V100'], allow_failed_imports=False)

# Hyper-Parameters
num_classes = 80  # 开放词汇总量
num_training_classes = 200  # 200高频
max_epochs = 2
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 1
load_from = "C:/Users/86198/Desktop/epoch_1.pth"

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
            model_name='D:/CLIP_checkpoints',
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

train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]
# ------------------------ VG 数据集配置 ------------------------

vg_train_dataset1 = dict(
    type='VGGroundingDataset',
    data_root='D:/',
    ann_file='D:/objects.json',
    img_info_file='D:/image_data.json',
    data_prefix=dict(img='VG_100K/'),
    filter_cfg=dict(
        filter_empty_gt=True,  # Flickr要求必须有标注
        min_size=32,
        max_gt_bbox=20),       # 限制每图最大标注数防止OOM
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=False,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=vg_train_dataset1)


# Validation Configuration (保持COCO验证)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='D:/COCO',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        batch_shapes_cfg=None),
    class_text_path='D:/YOLO-World-master/data/texts/coco_class_texts.json',
    pipeline=test_pipeline)

val_dataloader = dict(
    dataset=coco_val_dataset,
    batch_size=4,  # 验证批次大小
    num_workers=4
)
test_dataloader = val_dataloader

# ------------------------ 评估与训练策略 ------------------------
# Evaluation Metrics
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='D:/COCO/annotations/instances_val2017.json',
    metric='bbox',
    classwise=True,
    format_only=False  # 设为True可生成提交文件
)

test_evaluator = val_evaluator

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
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]

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
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),  # 文本编码器低学习率
            'logit_scale': dict(weight_decay=0.0)       # 对比学习温度参数
        }),
    constructor='YOLOWv5OptimizerConstructor',
    clip_grad=dict(max_norm=35, norm_type=2)  # 添加梯度裁剪
)