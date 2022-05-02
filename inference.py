"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd
from tqdm import tqdm

from evaluation import BoxEvaluator
from data_loaders import configure_metadata
from util import t2n

import torch
import matplotlib.pyplot as plt

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None, args=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.args = args

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        for images, targets, image_ids in tqdm(self.loader):
            image_size = images.shape[2:]
            images = images.cuda()
            outputs = self.model(images, labels=targets, return_cam=True)
            cams = t2n(outputs['cams'])

            for image, cam, image_id in zip(images.cpu().detach(), cams, image_ids):
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                if self.args.only_eval and self.args.save_cam:
                    image_orig = image * torch.tensor(_IMAGENET_STDDEV).view(3, 1, 1) + torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
                    name = image_id.replace('.' + image_id.split('.')[-1], '')
                    if self.split in ('val', 'test'):
                        cam_path = ospj(self.args.scoremap_paths[self.split], name)
                        if not os.path.exists(ospd(cam_path)):
                            os.makedirs(ospd(cam_path))
                        heatmap = cv2.applyColorMap(np.uint8(255*cam_normalized), cv2.COLORMAP_JET)
                        heatmap = np.float32(heatmap) / 255
                        heatmap = heatmap[...,::-1]
                        overlayed = 0.5 * image_orig.permute(1, 2, 0).numpy() + 0.3 * heatmap
                        plt.imsave(cam_path + '.png', overlayed, vmin=0.0, vmax=1.0)
                        np.save(ospj(cam_path), cam_normalized)
                self.evaluator.accumulate(cam_normalized, image_id)

        return self.evaluator.compute()