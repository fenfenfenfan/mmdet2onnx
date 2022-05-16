      
import cv2
import time
import numpy as np
import onnxruntime as ort

from custom import onehot2number,get_topk,py_cpu_nms,bbox2result,painter

img_path = "/mmdetection/coco/COCO_val2014_000000000488.jpg"
onnx_path = "/mmdetection/checkpoint/atss_coco_manu_no_topk_nms.onnx"
save_path = "workdir/save.jpg"
img_size = (608,608)    # resize
topk_threshold = 0.3    #topk的置信度阈值
topk = 100              #topk中的k
nms_iou_threshold = 0.6 #nms时iou阈值

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

sess = ort.InferenceSession(onnx_path)

# read pic 
img0 = cv2.imread(img_path)
# resize the pic
img1 = cv2.resize(img0, img_size)

scale_x,scale_y = img0.shape[1]/img1.shape[1],img0.shape[0]/img1.shape[0]
scale = [scale_x,scale_y,scale_x,scale_y,1]

# pre process
ori_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
input_image = ori_image.astype(np.float32) / 255.0
input_image = np.transpose(input_image, [2, 0, 1])
img_data = input_image[np.newaxis, :, :, :]

# using tool `Netron` to get the name of input node and output node
outputs = ["dets","labels"]

# inference
result = sess.run(outputs, {"input": img_data})


dets,labels = result
print("onnx直接输出:dets:",dets.shape,",labels:",labels.shape)

# post process
start_time = time.time()
# 返回det和labels，det中包含score
dets,labels = onehot2number(dets.squeeze(),labels.squeeze())
# 返回topk dets和labels
dets,labels = get_topk(dets,labels,topk_threshold,topk)
print("topk输出:dets:",dets.shape,",labels:",labels.shape)
# 按label序号顺序返回不同类别的bbox以及scores
dets_results = bbox2result(dets, labels, len(CLASSES))
# 
dets_results = [py_cpu_nms(result,nms_iou_threshold)*scale if len(result)!=0 else np.array([]).reshape((0,5)) for result in dets_results]
end_time = time.time()
print("Time used: ", end_time - start_time, 's')
# save
paint = painter(save_path,CLASSES,PALETTE)
paint.draw_bbox(img0,dets_results)

    