# dataset settings
data_source = 'CXR'
dataset_type = 'MultiViewDatasetCXR'
num_crops = [2, 6]
color_distort_strength = 1.0
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.14, 1.)),
    dict(type='RandomHorizontalFlip', p=0.5),
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=96, scale=(0.05, 0.14)),
    dict(type='RandomHorizontalFlip', p=0.5),
]

# prefetch
prefetch = False

# dataset summary
data = dict(
    samples_per_gpu=64,  # total 32*8=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        num_views=num_crops,
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch))
