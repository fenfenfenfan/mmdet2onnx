import torch
import json
import os
import os.path as osp
import mmcv
import argparse
import numpy as np
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmdet.datasets import (build_dataloader, build_dataset)
from mmcv.image import tensor2imgs
from PIL import Image


# checkpoint = "checkpoints/swa_model_12.pth"
# cfg_path = "configs/traffic/traffic.py"

# image_dir = "/task11"
# save_dir = "/task11_result/debuging"

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) an ONNX model using ONNXRuntime')
    parser.add_argument('--config', default='configs/atss/atss.py',help='test config file path')
    parser.add_argument('--model', default='checkpoints/atss_r50_fpn_1x_coco.onnx',help='Input model file')
    parser.add_argument('--print-file', help='output result file in pickle format')

    parser.add_argument(
        '--show-dir', default = 'work_dirs',help='directory where painted images will be saved')
    parser.add_argument(
        '--save-single', action='store_true',help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.1,
        help='score threshold (default: 0.3)')
    args = parser.parse_args()

    return args

def results2single(results, img_name, save_path):
    labels = ("pedestrian", "non-motor-vehicle", "motor-vehicle", "other")
    anno = {"info": {}, "annotations": []}
    anno["info"].update({"image_name": img_name})
    for result, label in zip(results[0], labels):
        result = result.tolist()
        for box in result:      
            w = box[2] - box[0]
            h = box[3] - box[1]
            anno["annotations"].append({
                "bbox": [(box[2] + box[0]) / 2, (box[3] + box[1]) / 2, w, h],
                "score":
                box[4],
                "category_name":
                label
            })
    with open(
            os.path.join(save_path, img_name[:-3] + 'json'), 'w',
            encoding='utf-8') as json_file:
        # print("anno save in {}".format(save_path))
        anno = json.dump(
            anno,
            json_file,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            separators=(',', ': '))

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)

    # build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)

    from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
    model = ONNXRuntimeDetector(
            args.model, class_names=dataset.CLASSES, device_id=0)

    model = MMDataParallel(model, device_ids=[0])

    print("onnx model detector inited!")
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        batch_size = len(result)

        if args.show_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                out_file = osp.join(args.show_dir, img_meta['ori_filename'])

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    out_file=out_file,
                    score_thr=args.show_score_thr)
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()

        img_name = data['img_metas'][0].data[0][0]['ori_filename']
        # save single image json result
        if args.save_single:
            single_save_dir = osp.join(args.show_dir,"single_result")
            if not osp.exists(single_save_dir):
                os.makedirs(single_save_dir)
            results2single(result, img_name, single_save_dir)

    print("finished!")


if __name__=="__main__":
    main()