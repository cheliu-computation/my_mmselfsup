# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import print_log

from .base import BaseDataset
from .builder import DATASETS
from .utils import to_numpy
import numpy as np

@DATASETS.register_module()
class SingleViewDatasetCXR(BaseDataset):
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
        super(SingleViewDatasetCXR, self).__init__(data_source, pipeline,
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

        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        
        label = np.zeros(10)
        return dict(img=img, label=label, idx=idx)

    def evaluate(self, results, logger=None, topk=(1, 5)):
        # """The evaluation function to output accuracy.

        # Args:
        #     results (dict): The key-value pair is the output head name and
        #         corresponding prediction values.
        #     logger (logging.Logger | str | None, optional): The defined logger
        #         to be used. Defaults to None.
        #     topk (tuple(int)): The output includes topk accuracy.
        # """
        # eval_res = {}
        # for name, val in results.items():
        #     val = torch.from_numpy(val)
        #     target = torch.LongTensor(self.data_source.get_gt_labels())
        #     assert val.size(0) == target.size(0), (
        #         f'Inconsistent length for results and labels, '
        #         f'{val.size(0)} vs {target.size(0)}')

        #     num = val.size(0)
        #     _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True)
        #     pred = pred.t()
        #     correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
        #     for k in topk:
        #         correct_k = correct[:k].contiguous().view(-1).float().sum(
        #             0).item()
        #         acc = correct_k * 100.0 / num
        #         eval_res[f'{name}_top{k}'] = acc
        #         if logger is not None and logger != 'silent':
        #             print_log(f'{name}_top{k}: {acc:.03f}', logger=logger)
        # return eval_res
        return NotImplemented