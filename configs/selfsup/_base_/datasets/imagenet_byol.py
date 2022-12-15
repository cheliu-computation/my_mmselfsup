# dataset settings
data_source = 'CXR'
dataset_type = 'MultiViewDatasetCXR'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline1 = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomResizedCrop', size=224, interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip')
]
train_pipeline2 = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomResizedCrop', size=224, interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False


# dataset summary
data = dict(
    samples_per_gpu=32,  # total 32*8(gpu)=256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch,
    ))
