# dataset settings
data_source = 'CXR'
# dataset_type = 'MultiViewDatasetCXR'
dataset_type_mimic = 'MultiViewDatasetMIMIC'
dataset_type_cxr14 = 'MultiViewDatasetNIH'
dataset_type_cxp = 'MultiViewDatasetCXP'
dataset_type_pdc = 'MultiViewDatasetPDC'
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
# data = dict(
#     samples_per_gpu=32,  # total 32*8(gpu)=256
#     workers_per_gpu=4,
#     train=dict(
#         type=dataset_type,
#         data_source=dict(
#             type=data_source,
#             data_prefix='cxr',
#             ann_file='cxr',
#         ),
#         num_views=[1, 1],
#         pipelines=[train_pipeline1, train_pipeline2],
#         prefetch=prefetch,
#     ))

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train = [
        dict(
        type=dataset_type_pdc,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch),
        dict(
        type=dataset_type_cxr14,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch),
        dict(
        type=dataset_type_cxp,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch),
        dict(
        type=dataset_type_mimic,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch)
    ]
    )
