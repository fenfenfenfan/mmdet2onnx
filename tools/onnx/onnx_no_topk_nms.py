import argparse
import numpy as np
import onnxruntime as ort
import cv2
import torch
from PIL import Image
#打印完整输出 
import numpy as np
np.set_printoptions(threshold=np.inf)
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) an ONNX model using ONNXRuntime')
    parser.add_argument('--image-path', default='/mmdetection/coco/COCO_train2014_000000000036.jpg',help='test config file path')
    parser.add_argument('--model', default='/mmdetection/checkpoint/atss_coco_manu_no_topk_nms.onnx',help='Input model file')
    parser.add_argument('--print-file', help='output result file in pickle format',default='./no_topk_nms.txt')

    parser.add_argument(
        '--show-dir', default = "./no_topk_nms.jpg",help='directory where painted images will be saved')
    parser.add_argument(
        '--save-single', action='store_true',help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.0,
        help='score threshold (default: 0.3)')
    args = parser.parse_args()

    return args

def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results



def main():
    args = parse_args()
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    
    # jpg格式RGB png格式RGBA
    img = Image.open(args.image_path)
    img1 = np.array(img)
    
    # 直接使用无后处理模型进行推理
    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 生成onnx用的是多大尺寸。这里就得是多大
    img = cv2.resize(img, (608, 608)) # np.array

    img = img.astype('float32')/225.0
    img = (img - mean)/ std
    image = img.transpose(2,0,1)  # w,h,c -> c,w,h
    image = image[None,:,:,:]
    image = np.array(image, dtype=np.float32)
    
    sess = ort.InferenceSession(args.model)
    raw_results = sess.run([],{'input':image})
    
    print("raw_results:\n",raw_results,file=open(args.print_file,'a'))
    print("len(raw_results):",len(raw_results),file=open(args.print_file,'a'))
    for i in range(len(raw_results)):
        print("{}:".format(i),raw_results[i].shape,file=open(args.print_file,'a'))

    pred_bboxes,pred_scores = raw_results
    
    # (1, 7706, 4)
    # (1, 7706, 80)
    pred_bboxes = pred_bboxes[0]
    pred_scores = pred_scores[0]

    
    # # show results
    # from mmdet.core.visualization import imshow_det_bboxes
    # imshow_det_bboxes(img = img1[:,:,::-1],bboxes=bboxes,labels=labels,class_names=CLASSES,
    #                     score_thr=args.show_score_thr,bbox_color=PALETTE,text_color=PALETTE,out_file=args.show_dir)


if __name__=="__main__":
    main()
    