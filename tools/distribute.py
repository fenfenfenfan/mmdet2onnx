# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os.path as osp
from mmcv import Config, mkdir_or_exist

from mmdet import __version__
from mmdet.datasets import build_dataset
from mmdet.utils import plotter


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)


    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])


    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))


    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    meta['exp_name'] = osp.basename(args.config)


    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # add an attribute for visualization convenience
    CLASSES = datasets[0].CLASSES
    distribution_plotter = plotter(len(CLASSES), cfg.work_dir)
    for ds in datasets[0]:
        c,h,w = ds["img"].data.shape
        bboxes = ds["gt_bboxes"].data
        labels = ds["gt_labels"].data
        distribution_plotter.update_bbox(h, 2/3, bboxes)
        distribution_plotter.update_cate(labels)
    distribution_plotter.draw_bbox_lower_bound()
    distribution_plotter.draw_categorical_distribution(True)
if __name__ == '__main__':
    main()
