      
      
import numpy as np
import os
from random import randint
import cv2
def onehot2number(boxes, scores):
    cls = np.argmax(scores,axis=1)
    scores = np.array([c[i] for i,c in zip(cls,scores)])
    dets = np.concatenate((boxes,scores.reshape(-1,1)),axis=1)
    return dets, cls

def get_topk(dets, cls, score_thr, topk):
    scores = dets[:,-1]
    valid_mask = scores > score_thr
    dets = dets[valid_mask]
    cls = cls[valid_mask]

    num_topk = min(topk, len(cls))
    ind = np.argsort(-scores[valid_mask])
    dets = dets[ind][:num_topk]
    cls = cls[ind][:num_topk]
    return dets, cls
    
 
def py_cpu_nms(dets, thresh):
    """
    nms
    :param dets: ndarray [x1,y1,x2,y2,score]
    :param thresh: int
    :return: list[index]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    order = dets[:, 4].argsort()[::-1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1] # 不包括第0个
    return dets[keep]
 


def bbox2result(bboxes, labels, num_classes):

    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        return [bboxes[labels == i, :] for i in range(num_classes)]

def xyxy2xxyy(bbox):
    for i in range(len(bbox)):
        x1 = int(bbox[i][0])
        y1 = int(bbox[i][1])
        x2 = int(bbox[i][2])
        y2 = int(bbox[i][3])
        bbox[i] = [x1,x2,y1,y2]
    return bbox

class painter():
    def __init__(self,save_path='../output/exp',CLASSES=[],PALETTE=None):
        self.save_path = save_path
        self.CLASSES = CLASSES
        if PALETTE is None:
            self.cols = [(randint(0,255),randint(0,255),randint(0,255)) for _ in range(len(CLASSES))]
        else:
            self.cols = PALETTE


    # draw single img
    def draw_bbox(self,img,dets):
        for label, det in enumerate(dets):
            bbox = det[:,:4].astype(int)
            score = det[:,-1]
            #bbox = xyxy2xxyy(bbox)
            px = 2
            for i,box in enumerate(bbox):
                cv2.rectangle(img, (box[0], box[1]), (box[2],box[3]), self.cols[label], px)
                self.__draw_rect(img,box,self.CLASSES[label],score[i],self.cols[label])
        cv2.imwrite(self.save_path,img)


    def __draw_rect(self,cvimg,box,cls,score,col):
        cls_len = len(cls)+5
        font_size = 1
        word_col = (255,255,255)
        cv2.rectangle(cvimg, (box[0]-1, box[1]-1), (box[0]+cls_len*10-1,box[1]-10-1), col, -1)
        cv2.putText(cvimg,cls+' '+str(score)[:4],(box[0]-1,box[1]+font_size-1),cv2.FONT_HERSHEY_PLAIN,font_size,word_col,1)

    