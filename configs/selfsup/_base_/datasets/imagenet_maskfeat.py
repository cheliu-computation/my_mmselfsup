# dataset settings
data_source = 'CXR'
dataset_type = 'SingleViewDatasetCXR'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
        size=224,
        scale=(0.5, 1.0),
        ratio=(0.75, 1.3333),
        interpolation='bicubic'),
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False

train_pipeline.append(dict(type='MaskFeatMaskGenerator', mask_ratio=0.4))

# dataset summary
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr'),
        pipeline=train_pipeline,
        prefetch=prefetch))
