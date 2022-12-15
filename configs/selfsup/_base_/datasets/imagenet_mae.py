# dataset settings
data_source = 'CXR'
# dataset_type = 'SingleViewDatasetCXR'
dataset_type_mimic = 'SingleViewDatasetMIMIC'
dataset_type_cxr14 = 'SingleViewDatasetCXR14'
dataset_type_cxp = 'SingleViewDatasetCXP'
dataset_type_pdc = 'SingleViewDatasetPDC'

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.2, 1.0), interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(type='Normalize', **img_norm_cfg),
]

# prefetch
prefetch = True
# if not prefetch:
#     train_pipeline.extend(
#         [dict(type='ToTensor'),
#          dict(type='Normalize', **img_norm_cfg)])

# dataset summary
# data_mimic = dict(
#     samples_per_gpu=256,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type_mimic,
#         data_source=dict(
#             type=data_source,
#             data_prefix='cxr',
#             ann_file='cxr',
#         ),
#         pipeline=train_pipeline,
#         prefetch=prefetch))
# data_cxp = dict(
#     samples_per_gpu=256,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type_cxp,
#         data_source=dict(
#             type=data_source,
#             data_prefix='cxr',
#             ann_file='cxr',
#         ),
#         pipeline=train_pipeline,
#         prefetch=prefetch))
# data_cxr14 = dict(
#     samples_per_gpu=256,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type_cxr14,
#         data_source=dict(
#             type=data_source,
#             data_prefix='cxr',
#             ann_file='cxr',
#         ),
#         pipeline=train_pipeline,
#         prefetch=prefetch))
# data_pdc = dict(
#     samples_per_gpu=256,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type_pdc,
#         data_source=dict(
#             type=data_source,
#             data_prefix='cxr',
#             ann_file='cxr',
#         ),
#         pipeline=train_pipeline,
#         prefetch=prefetch))

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=16,
    train = [
        dict(
        type=dataset_type_pdc,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch),
        dict(
        type=dataset_type_cxr14,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch),
        dict(
        type=dataset_type_cxp,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch),
        dict(
        type=dataset_type_mimic,
        data_source=dict(
            type=data_source,
            data_prefix='cxr',
            ann_file='cxr',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch)
    ]
    )
