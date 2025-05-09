_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# hyper-parameters
num_classes = 1203  # LVIS 数据集有 1203 个类别
num_training_classes = 80  # 训练时使用所有类别
max_epochs = 1
close_mosaic_epochs = 10  # 关闭 Mosaic 数据增强的轮数
save_epoch_intervals = 5  # 保存模型的间隔轮数
text_channels = 512  # 文本特征通道数
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]  # Neck 部分的嵌入通道数
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]  # Neck 部分的注意力头数
base_lr = 2e-4  # 初始学习率
weight_decay = 0.05  # 权重衰减
train_batch_size_per_gpu = 8  # 每个 GPU 的批量大小
load_from = '/root/autodl-tmp/tools/work_dirs/Pretrain_V100_test5/epoch_1.pth'  # 预训练权重
persistent_workers = False
# model settings
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
            model_name='/root/autodl-tmp/CLIP_checkpoints',  # CLIP 模型路径
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

# dataset settings
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

# LVIS 数据集配置
lvis_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='/root/autodl-tmp/COCO',
                 ann_file='/root/autodl-tmp/LVIS/lvis_v1_train.json',
                 data_prefix=dict(img='/root/autodl-tmp/COCO'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32)),
    class_text_path='/root/autodl-tmp/data/texts/lvis_v1_class_texts.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    #num_workers=0,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=lvis_train_dataset  # 使用 LVIS 数据集
)

# 验证集配置
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

lvis_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5LVISV1Dataset',
        data_root='/root/autodl-tmp/COCO',  # LVIS 数据集路径
        test_mode=True,
        ann_file='/root/autodl-tmp/LVIS/lvis_v1_minival_inserted_image_name.json',  # LVIS 验证集标注文件
        data_prefix=dict(img=''),  # 图像路径
        batch_shapes_cfg=None),
    class_text_path='/root/autodl-tmp/data/texts/lvis_v1_class_texts.json',  # LVIS 类别文本描述文件
    pipeline=test_pipeline)

val_dataloader = dict(dataset=lvis_val_dataset)
test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(
    _delete_=True,
    type='mmdet.LVISMetric',  # 使用 LVIS 评估器
    ann_file='/root/autodl-tmp/LVIS/lvis_v1_minival_inserted_image_name.json',  # LVIS 验证集标注文件
    metric='bbox'
)

test_evaluator = val_evaluator

# 训练设置
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best=None,
        interval=save_epoch_intervals))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=0,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')