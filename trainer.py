"""
Copyright (c) 2022 Eunji Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, for non-commercial use, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


================================================================================

This code is heavily borrowed from WSOL evaluation.
Therefore, this project contains subcomponents with separate copyright notices and license terms.

https://github.com/clovaai/wsolevaluation

WSOL evaluation

Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm

from data_loaders import get_data_loader
from inference import CAMComputer
from util import string_contains_any, normalize_minmax
import wsol
import wsol.method


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['fc.']
    }

    def __init__(self, args):
        self.args = args
        self.performance_meters = self._set_performance_meters()
        self.model = self._set_model()
        self.model_multi = torch.nn.DataParallel(self.model)
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.l1_loss = nn.L1Loss().cuda()
        self.optimizer = self._set_optimizer()
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.warm_batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)

    def reset_loaders(self):
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['loc_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if 'loss' in metric else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        arch = self.args.architecture
        model = wsol.__dict__[arch](
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            drop_threshold=self.args.drop_threshold,
            drop_prob=self.args.drop_prob)
        model = model.cuda()
        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []
        param_features_name = []
        param_classifiers_name = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture == 'vgg16':
                    param_features.append(parameter)
                    param_features_name.append(name)
                elif self.args.architecture == 'resnet50':
                    param_classifiers.append(parameter)
                    param_classifiers_name.append(name)
            else:
                if self.args.architecture == 'vgg16':
                    param_classifiers.append(parameter)
                    param_classifiers_name.append(name)
                elif self.args.architecture == 'resnet50':
                    param_features.append(parameter)
                    param_features_name.append(name)

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer

    def _get_loss_alignment(self, feature, sim, target, eps=1e-15):
        B = target.size(0)
        feature_norm = torch.norm(feature, dim=1)
        feature_norm_minmax = normalize_minmax(feature_norm)
        sim_target_flat = sim[torch.arange(B), target].view(B, -1)
        feature_norm_minmax_flat = feature_norm_minmax.view(B, -1)
        if self.args.dataset_name == 'ILSVRC':
            sim_fg = (feature_norm_minmax_flat > self.args.sim_fg_thres).float()
            sim_bg = (feature_norm_minmax_flat < self.args.sim_bg_thres).float()

            sim_fg_mean = (sim_fg * sim_target_flat).sum(dim=1) / (sim_fg.sum(dim=1) + eps)
            sim_bg_mean = (sim_bg * sim_target_flat).sum(dim=1) / (sim_bg.sum(dim=1) + eps)
            loss_sim = torch.mean(sim_bg_mean - sim_fg_mean)

            norm_fg = (sim_target_flat > 0).float()
            norm_bg = (sim_target_flat < 0).float()

            norm_fg_mean = (norm_fg * feature_norm_minmax_flat).sum(dim=1) / (norm_fg.sum(dim=1) + eps)
            norm_bg_mean = (norm_bg * feature_norm_minmax_flat).sum(dim=1) / (norm_bg.sum(dim=1) + eps)

            loss_norm = torch.mean(norm_bg_mean - norm_fg_mean)
        elif self.args.dataset_name == 'CUB':
            sim_fg = (feature_norm_minmax_flat > self.args.sim_fg_thres).float()
            sim_bg = (feature_norm_minmax_flat < self.args.sim_bg_thres).float()

            sim_fg_mean = (sim_fg * sim_target_flat).sum(dim=1) / (sim_fg.sum(dim=1) + eps)
            sim_bg_mean = (sim_bg * sim_target_flat).sum(dim=1) / (sim_bg.sum(dim=1) + eps)
            loss_sim = torch.mean(sim_bg_mean - sim_fg_mean)

            sim_max_class, _ = sim.max(dim=1)
            sim_max_class_flat = sim_max_class.view(B, -1)

            norm_fg = (sim_max_class_flat > 0).float()
            norm_bg = (sim_max_class_flat < 0).float()

            norm_fg_mean = (norm_fg * feature_norm_minmax_flat).sum(dim=1) / (norm_fg.sum(dim=1) + eps)
            norm_bg_mean = (norm_bg * feature_norm_minmax_flat).sum(dim=1) / (norm_bg.sum(dim=1) + eps)

            loss_norm = torch.mean(norm_bg_mean - norm_fg_mean)
        else:
            raise ValueError("dataset_name should be in ['ILSVRC', 'CUB']")

        return loss_sim, loss_norm

    def _wsol_training(self, images, target, warm=False):
        output_dict = self.model_multi(images, labels=target)
        logits = output_dict['logits']

        if self.args.wsol_method == 'bridging-gap':
            loss_ce = self.cross_entropy_loss(logits, target)

            loss_drop = self.l1_loss(output_dict['feature'], output_dict['feature_erased'])
            loss_sim, loss_norm = \
                self._get_loss_alignment(output_dict['feature'], output_dict['sim'], target)

            loss = loss_ce + self.args.loss_ratio_drop * loss_drop
            if not warm:
                loss += self.args.loss_ratio_sim * loss_sim + self.args.loss_ratio_norm * loss_norm
        elif self.args.wsol_method == 'cam':
            loss = self.cross_entropy_loss(logits, target)
        else:
            raise ValueError("wsol_method should be in ['bridging-gap', 'cam']")

        return logits, loss

    def train(self, split, warm=False):
        self.model_multi.train()
        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, _) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

            logits, loss = self._wsol_training(images, target, warm=warm)
            pred = logits.argmax(dim=1)
            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        self.performance_meters[split]['classification'].update(classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc, loss=loss_average)

    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                scale = '%' if metric != 'loss' else ''
                current_performance = self.performance_meters[split][metric].current_value

                if current_performance is not None and not ('loss' in metric and current_performance == np.inf)\
                        and not ('loss' not in metric and current_performance < 0):
                    best_message = "\t(best value:\t{0:0.3f}{1},\tat epoch {2})".format(
                        self.performance_meters[split][metric].best_value,
                        scale,
                        self.performance_meters[split][metric].best_epoch)

                    print("Split {0},\tmetric {1},\tcurrent value:\t{2:0.3f}{3}{4}".format(
                        split, metric, current_performance, scale, best_message))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_accuracy(self, loader, topk=(1,)):
        num_correct = np.zeros((len(topk)))
        num_images = 0

        for i, (images, targets, image_ids) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()
            output_dict = self.model_multi(images)
            pred_topk = torch.argsort(output_dict['logits'], dim=1, descending=True)[:, :max(topk)]

            for i_k, k in enumerate(topk):
                co = (pred_topk[:, :k] == targets.unsqueeze(-1)).float()
                num_correct[i_k] += torch.sum(torch.max(co, dim=1)[0]).item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100

        return classification_acc

    def evaluate(self, epoch, split):
        print("Evaluate epoch {}, split {}".format(epoch, split))
        self.model_multi.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy[0])

        cam_computer = CAMComputer(
            model=self.model_multi,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            log_folder=self.args.log_folder,
            args=self.args
        )

        cam_performance = cam_computer.compute_and_evaluate_cams()

        if self.args.multi_iou_eval:
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[split]['localization'].update(loc_score)

        if self.args.dataset_name in ('CUB', 'ILSVRC'):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split]['loc_IOU_{}'.format(IOU_THRESHOLD)].update(cam_performance[idx])

    def evaluate_empty(self, split):
        for metric in list(self.performance_meters[split].keys()):
            self.performance_meters[split][metric].update(None)

    def _torch_save_model(self, filename):
        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split='val'):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('best'))
        self._torch_save_model(
            self._CHECKPOINT_NAME_TEMPLATE.format('last'))

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch in self.args.lr_decay_points:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.args.lr_decay_rate

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.higher_is_better = higher_is_better
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split in ['val', 'test'] else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value=None):
        if new_value is None:
            new_value = -np.inf if self.higher_is_better else np.inf
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)
