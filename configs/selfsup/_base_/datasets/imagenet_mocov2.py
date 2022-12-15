# dataset settings
data_source = 'CXR'
dataset_type = 'MultiViewDatasetCXR'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# The difference between mocov2 and mocov1 is the transforms in the pipeline
train_pipeline = [
    # dict(type='ToTensor'),
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomHorizontalFlip'),
]

# prefetch
prefetch = False
# if not prefetch:
#     train_pipeline.extend(
#         [dict(type='ToTensor'),
#          dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=300,  # total 32*8=256
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))
