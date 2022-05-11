_base_ = './traffic.py'

model = dict(
    bbox_head=dict(num_classes=9))

dataset_type = 'KittiDataset'
data_root = 'data/kitti2d/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'training/kitti2d_train.json',
        img_prefix=data_root + 'training/image_2/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'training/kitti2d_val.json',
        img_prefix=data_root + 'training/image_2/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'training/kitti2d_val.json',
        img_prefix=data_root + 'training/image_2/'))