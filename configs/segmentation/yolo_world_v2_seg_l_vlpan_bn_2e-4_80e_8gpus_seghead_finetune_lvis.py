

_base_ = (
    '../../third_party/mmyolo/configs/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py'
)

custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)
# hyper-parameters
num_classes = 1203
num_training_classes = 80
max_epochs = 1  # Maximum training epochs
close_mosaic_epochs = 0
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 4
load_from = '/root/autodl-tmp/tools/work_dirs/yolo_world_v2_seg_l_vlpan_bn_2e-4_80e_8gpus_seghead_finetune_lvis/epoch_1.pth'
# text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
#text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = False

# 数据增强简化配置
downsample_ratio = 4
mask_overlap = False
use_mask2refine = True
max_aspect_ratio = 100
min_area_ratio = 0.01

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
            model_name='/root/autodl-tmp/CLIP_checkpoints',
            frozen_modules=[])),#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    neck=dict(
        type='YOLOWorldPAFPN',
        freeze_all=False,
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldSegHead',
                   head_module=dict(type='YOLOWorldSegHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes,
                                    mask_channels=32,
                                    proto_channels=256),
                                    #freeze_bbox=True),
                   mask_overlap=mask_overlap,
                   loss_mask=dict(type='mmdet.CrossEntropyLoss',
                                  use_sigmoid=True,
                                  reduction='none'),
                   loss_mask_weight=1.0),
    train_cfg=dict(
        assigner=dict(
            type='YOLOWorldSegAssigner',
            num_classes=num_training_classes,

        )
    ),
    test_cfg=dict(
        # 仅保留分割相关测试配置
        mask_thr_binary=0.5,
        #max_per_img=100,
        #mask_resolution=256,
        fast_test=True,

    )
)


pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         mask2bbox=True)
]

# 简化后的last_transform
last_transform = [
    dict(type='YOLOv5HSVRandomAug',
         hue_delta=5,  # 减小参数
         saturation_delta=30,
         value_delta=30),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='Polygon2Mask',
         downsample_ratio=downsample_ratio,
         mask_overlap=mask_overlap),
]
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
# 简化后的mosaic_affine_transform
mosaic_affine_transform = [
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=pre_transform),
    dict(type='YOLOv5CopyPaste', prob=_base_.copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=True)
]
train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(type='YOLOv5MultiModalMixUp',
         prob=_base_.mixup_prob,
         pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform, *text_transform
]

_train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(type='LetterResize',
         scale=_base_.img_scale,
         allow_scale_up=True,
         pad_val=dict(img=114.0)),
    dict(type='YOLOv5RandomAffine',
         max_rotate_degree=0.0,
         max_shear_degree=0.0,
         scaling_ratio_range=(1 - _base_.affine_scale,
                              1 + _base_.affine_scale),
         max_aspect_ratio=_base_.max_aspect_ratio,
         border_val=(114, 114, 114),
         min_area_ratio=min_area_ratio,
         use_mask_refine=use_mask2refine), *last_transform
]
train_pipeline_stage2 = [*_train_pipeline_stage2, *text_transform]
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='/root/autodl-tmp/COCO',
                 ann_file='/root/autodl-tmp/LVIS/lvis_v1_train.json',
                 data_prefix=dict(img='/root/autodl-tmp/COCO'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32)),
    class_text_path='/root/autodl-tmp/data/texts/lvis_v1_class_texts.json',
    pipeline=train_pipeline)
train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)


test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=2,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(#bias_decay_mult=0.0,
                                        #norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'neck':
                                            dict(lr_mult=1),
                                            'head':
                                            dict(lr_mult=1)

                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='/root/autodl-tmp/COCO',
                 test_mode=True,
                 ann_file='/root/autodl-tmp/LVIS/lvis_v1_minival_inserted_image_name.json',
                 data_prefix=dict(img=''),
                 batch_shapes_cfg=None),
    class_text_path='/root/autodl-tmp/data/texts/lvis_v1_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='/root/autodl-tmp/LVIS/lvis_v1_minival_inserted_image_name.json',
                     metric=['bbox', 'segm'])

test_evaluator = val_evaluator
find_unused_parameters = True
