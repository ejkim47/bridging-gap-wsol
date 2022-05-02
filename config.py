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

import argparse
import munch
import os
from os.path import join as ospj
import shutil
import warnings
import yaml
from util import Logger

_DATASET_NAMES = ('CUB', 'ILSVRC')
_ARCHITECTURE_NAMES = ('vgg16', 'resnet50')
_METHOD_NAMES = ('cam', 'bridging-gap')
_SPLITS = ('train', 'val', 'test')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def box_v2_metric(args):
    if args.box_v2_metric:
        args.multi_contour_eval = True
        args.multi_iou_eval = True
    else:
        args.multi_contour_eval = False
        args.multi_iou_eval = False
        warnings.warn("MaxBoxAcc metric is deprecated.")
        warnings.warn("Use MaxBoxAccV2 by setting args.box_v2_metric to True.")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_architecture_type(wsol_method):
    if wsol_method == 'bridging-gap':
        architecture_type = 'drop'
    else:
        architecture_type = 'cam'
    return architecture_type


def configure_data_paths(args):
    train = val = test = ospj(args.data_root, args.dataset_name)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def configure_scoremap_output_paths(args):
    scoremaps = None
    if args.save_cam:
        scoremaps_root = ospj(args.log_folder, 'scoremaps')
        scoremaps = mch()
        for split in ('val', 'test'):
            scoremaps[split] = ospj(scoremaps_root, split)
            if not os.path.isdir(scoremaps[split]):
                os.makedirs(scoremaps[split])
    return scoremaps


def configure_log_folder(args):
    if args.only_eval:
        log_folder = args.config.replace(args.config.split('/')[-1], '')
        assert ('config' not in log_folder)
    else:
        log_folder = ospj(args.log_dir, '{}_{}_{}'.format(args.dataset_name, args.architecture, args.experiment_name))

    if args.only_eval:
        print(log_folder)
        assert (os.path.isdir(log_folder))
        print("Inference with last_checkpoint and best_checkpoint in", log_folder)
    elif os.path.isdir(log_folder):
        if args.override_cache:
            shutil.rmtree(log_folder, ignore_errors=True)
            os.makedirs(log_folder)
        else:
            raise RuntimeError("Experiment with the same name exists: {}"
                               .format(log_folder))
    else:
        os.makedirs(log_folder)
    return log_folder


def configure_log(args):
    if not args.only_eval:
        log_file_name = ospj(args.log_folder, 'log.log')
        Logger(log_file_name)


def configure_config(args):
    if not args.only_eval:
        shutil.copy(src=os.path.join(os.getcwd(), args.config), dst=args.log_folder)


def configure_lr_decay(args):
    if args.lr_decay_points is None:
        assert args.lr_decay_frequency != 0
        lr_decay_points = [e for e in range(1, args.epochs + 1) if e % args.lr_decay_frequency == 0]
    else:
        lr_decay_points = args.lr_decay_points
    return lr_decay_points


def check_dependency(args):
    if args.dataset_name == 'CUB':
        if args.num_val_sample_per_class >= 6:
            raise ValueError("num-val-sample must be <= 5 for CUB.")


def get_configs():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--config', type=str, default='configs/config_exp.yaml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_cam', action='store_true', default=False)
    parser.add_argument('--only_eval', action='store_true', default=False)

    flags = parser.parse_args()

    with open(flags.config, 'rb') as f:
        conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    conf.update(vars(flags))
    args = argparse.Namespace(**conf)

    check_dependency(args)
    args.log_folder = configure_log_folder(args)
    configure_log(args)
    configure_config(args)
    box_v2_metric(args)

    args.architecture_type = get_architecture_type(args.wsol_method)
    args.data_paths = configure_data_paths(args)
    args.metadata_root = ospj(args.metadata_root, args.dataset_name)
    args.scoremap_paths = configure_scoremap_output_paths(args)
    args.lr_decay_points = configure_lr_decay(args)

    return args
