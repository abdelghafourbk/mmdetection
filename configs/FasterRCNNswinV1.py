# the new config inherits the base configs to highlight the necessary modification
_base_ = 'swin/faster-rcnn_swin.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('meteor','nonmeteor')
data_root='/kaggle/input/datameteors/datamet'
backend_args = None

train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile', backend_args=backend_args),  # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_mask=False,  # Whether to use instance mask, True for instance segmentation
        poly2mask=False),  # Whether to convert the polygon mask to instance mask, set False for acceleration and to save memory
    dict(type='PackDetInputs')  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train'),
        pipeline=train_pipeline)
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='validation.json',
        data_prefix=dict(img='valid')
        )
    )

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + '/validation.json',  # Annotation file path
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection
    iou_thrs = [0.3,0.35,0.4,0.45,0.5],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator  # Testing evaluator config
'''
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotation_data',
        data_prefix=dict(img='test/image_data')
        )
    )
'''
# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=2,
                dropout=0.1)))

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

from mmcv.runner.hooks.lr_updater import ReduceLROnPlateauHook

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=max_epochs,
    #     by_epoch=True,
    #     milestones=[8, 11],
    #     gamma=0.1)
]

# Add ReduceLROnPlateauHook to your training workflow
lr_config = dict(
    policy='step',  # Policy for adjusting learning rate. Options: 'step', 'linear', 'exp', 'cosine', 'poly'
    warmup='linear',  # Policy for warmup. Options: 'linear', 'constant'
    warmup_iters=500,  # Number of iterations for warmup
    warmup_ratio=0.001,  # Ratio of starting learning rate used in warmup
    step=[8, 11],  # List of epochs to adjust learning rate
    gamma=0.1,  # Factor by which to reduce the learning rate
    by_epoch=True  # Whether to update learning rate by epoch or by iteration
)
# Add ReduceLROnPlateauHook to the training pipeline
hooks = [
    dict(type='CheckpointHook', interval=1),
    dict(type='IterTimerHook'),
    dict(type='LrUpdaterHook'),
    dict(type='EvalHook', interval=1),
    dict(type='TextLoggerHook'),
    # Add ReduceLROnPlateauHook
    dict(
        type='ReduceLROnPlateauHook',
        by_epoch=True,
        **lr_config
    )
]


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.0001))
