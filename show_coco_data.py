"""
visualize coco format data
you can choose the num image to show 
e.g python /mmdetection/z_my_test_tools/show_coco_data.py --show-single --show-num 50
you also can choose the particular category to show, this will show all the images contain the paiticular category
e.g python /mmdetection/z_my_test_tools/show_coco_data.py --show-particular --category-id 2
use mmdetection utils
"""
import argparse
import random
from tqdm import tqdm
from PIL import Image
import json
import os
import numpy as np
from torchvision.transforms import functional as F
# 解决相对导入问题
from mmdet.core.visualization import imshow_det_bboxes

# print(__name__)
# print(__package__)
# from ..mmdet.core.visualization import imshow_det_bboxes

data_root = '/mmdetection/data/traffic/task11_origin/data/'
annos_path = '/mmdetection/data/ali_tianchi/train/annotations/instances_train2017.json'
imgs_root = '/mmdetection/data/ali_tianchi/train/images'
out_dir1 = './test_show_data/single'
out_dir2 = './test_show_data/particular'
if not os.path.exists(out_dir1):
    os.makedirs(out_dir1)
if not os.path.exists(out_dir2):
    os.makedirs(out_dir2)


def parse_args():
    parser = argparse.ArgumentParser(description='show bbox')
    parser.add_argument('--show-single', action='store_true')
    parser.add_argument('--show-particular', action='store_true')
    parser.add_argument('--show-num', type=int, default=10)
    parser.add_argument('--category-id', type=int, default=10)

    args = parser.parse_args()

    if args.show_num:
        assert args.show_single is not None
    if args.category_id:
        assert args.show_particular is not None
    return args


def get_single_anno(data_dict, image_id):
    # img_id 0-n
    imgs_path = [
        os.path.join(imgs_root, img['file_name'])
        for img in data_dict['images']
    ]
    # print(image_id)
    # json文件中如果image_id从1开始，则image_id作为索引需要减1（task11）
    # json文件中如果image_id从0开始，则image_id作为索引不需要减1（ali）
    img_path = imgs_path[image_id]
    anno = data_dict['annotations']
    gt_bboxes = []
    gt_labels = []
    for obj in anno:
        if obj["image_id"] == image_id:
            x1 = int(obj["bbox"][0])
            y1 = int(obj["bbox"][1])
            x2 = int(obj["bbox"][0] + obj["bbox"][2])
            y2 = int(obj["bbox"][1] + obj["bbox"][3])
            gt_bboxes.append([x1, y1, x2, y2])
            # 后面画图工具是用label去索引，所以需要减一
            gt_labels.append(obj["category_id"] - 1)

    return img_path, gt_bboxes, gt_labels


def get_particular_anno(data_dict, category_id):
    """find all image_id which contain particular category"""
    anno = data_dict['annotations']
    images_id = []
    results = []
    for obj in anno:
        if obj["category_id"] == category_id and obj[
                "image_id"] not in images_id:
            images_id.append(obj["image_id"])
    for image_id in images_id:
        results.append(get_single_anno(data_dict, image_id))

    return results


def show_image(ima, bbox, label):
    pass


def main():
    args = parse_args()
    with open(annos_path, 'r') as f:
        data_dict = json.load(f)
    class_names = [
        class_name["name"] for class_name in data_dict['categories']
    ]
    print(class_names)
    print('num classes: ', len(class_names))
    imgs_name = [img['file_name'] for img in data_dict['images']]
    length = len(imgs_name)
    print("{} images".format(length))
    idx = random.sample(range(0, length), args.show_num)

    if args.show_single:
        for i in tqdm(idx):
            # print(i)
            # with Image.open(imgs_path[i]) as f:
            #     img = F.to_tensor(f)
            img_path, gt_bboxes, gt_labels = get_single_anno(data_dict, i)
            img = img_path
            bbox = np.array(gt_bboxes)
            label = np.array(gt_labels)
            # label = np.expand_dims(label,1)
            out_file = os.path.join(out_dir1,
                                    "result_" + os.path.basename(img))
            imshow_det_bboxes(
                img=img,
                bboxes=bbox,
                labels=label,
                # segms=annotation[i]["gt_seg"],
                class_names=class_names,
                out_file=out_file)
    elif args.show_particular:
        results = get_particular_anno(data_dict, args.category_id)
        for img_path, gt_bboxes, gt_labels in tqdm(results):
            img = img_path
            bbox = np.array(gt_bboxes)
            label = np.array(gt_labels)
            # label = np.expand_dims(label,1)
            out_file = os.path.join(out_dir2,
                                    "result_" + os.path.basename(img))
            imshow_det_bboxes(
                img=img,
                bboxes=bbox,
                labels=label,
                # segms=annotation[i]["gt_seg"],
                class_names=class_names,
                out_file=out_file)


if __name__ == "__main__":
    print('start')
    main()
    print('end')