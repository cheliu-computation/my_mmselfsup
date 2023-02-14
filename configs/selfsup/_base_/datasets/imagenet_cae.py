# dataset settings
data_source = 'CXR'
# dataset_type = 'SingleViewDatasetCXR'
dataset_type_mimic = 'SingleViewDatasetMIMIC'
dataset_type_cxr14 = 'SingleViewDatasetNIH'
dataset_type_cxp = 'SingleViewDatasetCXP'
dataset_type_pdc = 'SingleViewDatasetPDC'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
        size=224,
        second_size=112,
        interpolation='bicubic',
        second_interpolation='lanczos',
        scale=(0.08, 1.0)),
]

# prefetch
prefetch = True
# if not prefetch:
#     train_pipeline.extend([dict(type='ToTensor')])

train_pipeline.append(
    dict(
        type='BEiTMaskGenerator',
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=None,
        min_num_patches=16))

# dataset summary
# data = dict(
#     samples_per_gpu=256,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type,
#         data_source=dict(
#             type=data_source,
#             data_prefix='data/imagenet/train',
#             ann_file='data/imagenet/meta/train.txt'),
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
