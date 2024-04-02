
# the new config inherits the base configs to highlight the necessary modification
_base_ = 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('__background__', 'meteor')
data_root='/kaggle/input/datameteors/meteor'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='coco_train',
        data_prefix=dict(img='frames')
        )
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
        ann_file='coco_val',
        data_prefix=dict(img='frames')
        )
    )
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
                num_classes=2)))
