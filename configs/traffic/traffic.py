# model settings
fp16 = dict(loss_scale=dict(init_scale=512))
model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        # regress_ranges=((-1, 16), (16, 32), (32, 64), (64, 128), (128, 1e8)),
        center_sampling=False,
        dcn_on_last_conv=True,
        use_atss=True,
        use_vfl=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            # 8->6.5
            octave_base_scale=6.5,
            scales_per_octave=1,
            center_offset=0.0,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.0001),
        max_per_img=300))

# dataset settings
dataset_type = 'TrafficDataset'
data_root = 'data/traffic/task11_origin/data/'
all_data_root = 'data/traffic/task11/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(2000, 1176), (2000, 704)],
        multiscale_mode='range',
        keep_ratio=True),
    # dict(type='Resize', img_scale=(2000, 1216), keep_ratio=True),
    dict(
        type='RandomFlip',
        flip_ratio=0.5),
    dict(type='AutoAugment', autoaug_type='v1'),
    dict(type='GridMask',mode=1,prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2000, 1176),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip',
                # direction=['horizontal', 'vertical']
            ),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=all_data_root + 'annotations/train.json',
        img_prefix=all_data_root + 'train_images/',
        # ann_file=data_root + 'annotations/train.json',
        # img_prefix=data_root + 'train_images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/val.json',
        # img_prefix=data_root + 'val_images/',
        ann_file='task11_annotation/task11.json',
        img_prefix='/task11/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/val.json',
        # img_prefix=data_root + 'val_images/',
        ann_file='task11_annotation/task11.json',
        img_prefix='/task11/',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metric='bbox',start=12)

# schedules
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.00125*12,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [
    dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/from_kitti.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
