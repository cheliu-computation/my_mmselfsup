# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import print_log

from .base import BaseDataset
from .builder import DATASETS
from .utils import to_numpy
import numpy as np

@DATASETS.register_module()
class SingleViewDatasetMIMIC(BaseDataset):
    """The dataset outputs one view of an image, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(SingleViewDatasetMIMIC, self).__init__(data_source, pipeline,
                                                prefetch)

        self.data_source = np.load('/raid/cl522/XSCAN/all_data/MIMIC.npy', 'r')

        self.prefetch = prefetch

    def __getitem__(self, idx):
        img = self.data_source[idx]

        img = np.array(img, dtype=np.float)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        if len(img.shape)>2:
            img = img[:,:,0]
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = torch.tensor(img).to(torch.float32)
        img = torch.permute(img, (2, 0, 1))

        # if self.prefetch:
        #     img = torch.from_numpy(to_numpy(img))
        
        label = np.zeros(10)
        return dict(img=img, label=label, idx=idx)

    def evaluate(self, results, logger=None, topk=(1, 5)):
        return NotImplemented

@DATASETS.register_module()
class SingleViewDatasetNIH(BaseDataset):
    """The dataset outputs one view of an image, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(SingleViewDatasetNIH, self).__init__(data_source, pipeline,
                                                prefetch)

        self.data_source = np.load('/raid/cl522/XSCAN/all_data/CXR14.npy', 'r')

        self.prefetch = prefetch

    def __getitem__(self, idx):
        img = self.data_source[idx]

        img = np.array(img, dtype=np.float)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        if len(img.shape)>2:
            img = img[:,:,0]
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = torch.tensor(img).to(torch.float32)
        img = torch.permute(img, (2, 0, 1))

        # if self.prefetch:
        #     img = torch.from_numpy(to_numpy(img))
        
        label = np.zeros(10)
        return dict(img=img, label=label, idx=idx)

    def evaluate(self, results, logger=None, topk=(1, 5)):
        return NotImplemented

@DATASETS.register_module()
class SingleViewDatasetCXP(BaseDataset):
    """The dataset outputs one view of an image, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(SingleViewDatasetCXP, self).__init__(data_source, pipeline,
                                                prefetch)

        self.data_source = np.load('/raid/cl522/XSCAN/all_data/CXP.npy', 'r')

        self.prefetch = prefetch

    def __getitem__(self, idx):
        img = self.data_source[idx]

        img = np.array(img, dtype=np.float)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        if len(img.shape)>2:
            img = img[:,:,0]
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = torch.tensor(img).to(torch.float32)
        img = torch.permute(img, (2, 0, 1))

        # if self.prefetch:
        #     img = torch.from_numpy(to_numpy(img))
        
        label = np.zeros(10)
        return dict(img=img, label=label, idx=idx)

    def evaluate(self, results, logger=None, topk=(1, 5)):
        return NotImplemented

@DATASETS.register_module()
class SingleViewDatasetPDC(BaseDataset):
    """The dataset outputs one view of an image, containing some other
    information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(SingleViewDatasetPDC, self).__init__(data_source, pipeline,
                                                prefetch)

        self.data_source = np.load('/raid/cl522/XSCAN/PDC_new.npy', 'r')

        self.prefetch = prefetch

    def __getitem__(self, idx):
        img = self.data_source[idx]

        img = np.array(img, dtype=np.float)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        if len(img.shape)>2:
            img = img[:,:,0]
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = torch.tensor(img).to(torch.float32)
        img = torch.permute(img, (2, 0, 1))

        # if self.prefetch:
        #     img = torch.from_numpy(to_numpy(img))
        
        label = np.zeros(10)
        return dict(img=img, label=label, idx=idx)

    def evaluate(self, results, logger=None, topk=(1, 5)):
        return NotImplemented