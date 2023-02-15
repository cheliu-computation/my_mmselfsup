# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
# python extract_backbone_weights.py /home/cl522/github_repo/res50_allCXR_log/byol/latest.pth /home/cl522/github_repo/res50_allCXR_log/BYOL_pre_backbone.pth

def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith('.pth')
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author='MMSelfSup')
    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
            has_backbone = True
    if not has_backbone:
        raise Exception('Cannot find a backbone module in the checkpoint.')
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
