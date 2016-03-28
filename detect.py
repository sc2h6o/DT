import cv2
import os
import sys
sys.path.append("/home/syc/caffe-new/python")
import caffe
import numpy as np
import random
import time
from math import *
from DataBase import DataBase

video_dir = "../../dataset/"
video_name = "woman"
video_transpose = False
video_resize = (960, 540)
bbox = 207,117,29,103
# bbox = 100,20,60,60
(x,y,w,h) = bbox

batch_size = 4

from_seq = True

scale = 1

model_dir = 'model/'
data_dir = 'data/'
proto_file = model_dir + 'solver.prototxt'
model_file =  model_dir + "ZF_faster_rcnn_final.caffemodel"
mean_file = model_dir + 'ilsvrc_2012_mean.npy'

target_size = 224

class DeepTracker:
    def __init__(self, proto_file, model_file, mean_file):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.inited = False
        self.prob = None
        self.mean = np.array([102.9801, 115.9465, 122.7717])

    def transpose(self, frame, bbox):
        (x,y,w,h) = bbox
        pad = 200
        _frame = np.zeros((frame.shape[0] + 2*pad, frame.shape[0] + 2*pad, 3))
        _frame[pad:pad+frame.shape[0], pad:pad+frame.shape[1], :] = frame
        data = _frame[pad+y:pad+y+h, pad+x:pad+x+w, :]
        # data = cv2.resize(data, (scale*w,scale*h))
        data = data - self.mean
        data = data.transpose((2,0,1)) 
        return data


    def transform(self, bbox, delta=(0,0,-0,-0)):
        (x,y,w,h) = bbox
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        w = w / exp(delta[2])
        h = h / exp(delta[3])
        cx -= delta[0] * w
        cy -= delta[1] * h
        x = cx - 0.5 * w
        y = cy - 0.5 * h
        return (int(round(x)),int(round(y)),int(round(w)),int(round(h)))

    def transform_inv(self, bbox, delta):
        (x,y,w,h) = bbox
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        cx += delta[0] * w
        cy += delta[1] * h
        w = w * exp(delta[2])
        h = h * exp(delta[3])
        x = cx - 0.5 * w
        y = cy - 0.5 * h
        return (int(round(x)),int(round(y)),int(round(w)),int(round(h)))


    def IoU(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0]+box1[2], box2[0]+box2[2])
        y2 = min(box1[1]+box1[3], box2[1]+box2[3])
        if x1>=x2 or y1>=y2:
            return 0
        else:
            a1 = box1[2] * box2[3]
            a2 = box1[2] * box2[3]
            return (x2-x1)*(y2-y1)*1.0 / (a1 + a2)

    def saveImage(self, frame, bbox, idx):
        x,y,w,h = bbox
        cv2.imwrite('data/%d.jpg'%idx, frame[y:y+h,x:x+w,:])

    def genData(self, frame, bbox, dw , dh):
        num_pos = 0
        for i in range(batch_size):
            rand_range = 0.2
            dx = rand_range * (2*random.random()-1)
            dy = rand_range * (2*random.random()-1)
            delta = np.array([dx,dy,dw,dh])
            box_sample = self.transform(bbox, delta)
            # self.saveImage(frame,box_sample,i)
            box_large = self.transform(bbox)
            # data_sample = self.transpose(frame, box_sample)
            IoU = self.IoU(box_sample, box_large)
            if IoU > 0.1:
                label = 1
                num_pos += 1
            else:
                label = 0

            x,y,w,h = box_sample
            roi = np.array([0,x,w,x+w,y+h])
            self.db.insertSample(label, roi, delta)


    def update(self, frame, bbox, batch_num = 32, loss_cls=0.001, loss_bbox=0.001):
        t1 = time.clock()
        x,y,w,h = bbox
        for i in range(batch_num):
            dw = 0
            dh = 0
            _w = int(round(w / exp(dw)))
            _h = int(round(h / exp(dh)))
            data = self.transpose(frame, (0,0,frame.shape[1],frame.shape[0]))
            self.db.initData(batch_size, frame.shape[1], frame.shape[0], data)
            self.genData(frame, bbox, dw, dh)
            self.solver.step(1)

        t2 = time.clock() - t1
        print 'update takes %f second.' % (t2-t1)

    def init(self, frame, bbox):
        self.solver = caffe.SGDSolver(proto_file)
        self.solver.net.copy_from(model_file)
        self.db = DataBase(self.solver.net)

        self.update(frame, bbox, 1024)
        
        self.inited = True
        self.prob = np.zeros((frame.shape[0],frame.shape[1]))

    def track(self, frame, bbox):
        (x,y,w,h) = self.transform(bbox)
        box_large = (x,y,w,h)
        data = self.transpose(frame, (0,0,frame.shape[1],frame.shape[0]))
        self.solver.net.blobs['data'].reshape(1,3,frame.shape[0],frame.shape[1])
        self.solver.net.blobs['data'].data[0] = data
        self.solver.net.blobs['rois'].data[0] = np.array([0,x,y,x+w,y+h])
        # self.solver.net.blobs['label'].data = np.array([1])
        # self.solver.net.blobs['bbox_inside_weights'].data = np.array([10,5,1,1])
        # self.solver.net.blobs['bbox_outside_weights'].data = np.array([10,5,1,1])
        self.solver.net.forward()

        scores = self.solver.net.blobs['cls_score'].data[0]
        box_deltas = self.solver.net.blobs['bbox_pred'].data[0]
        box_deltas[2] = box_deltas[3] = 0
        pred_box = self.transform_inv(box_large, box_deltas)
        print box_deltas

        # self.update(frame, bbox)
        return pred_box



if __name__ == "__main__":
    
    dt = DeepTracker(proto_file, model_file, mean_file)

    success, frame = True, None
    seq = []
    idx = 0
    if from_seq:
        for filename in os.listdir(os.path.join(video_dir,video_name)):
            if '.jpg' in filename:
                seq.append(os.path.join(video_dir,video_name,filename))
        seq.sort()
        frame = cv2.imread(seq[idx])
        idx += 1
    else:
        cap = cv2.VideoCapture(video_dir+video_name)
        success, frame = cap.read()
    while success :
        t1 = time.clock()
        
        if dt.inited:
            bbox = dt.track(frame, bbox)
            # cv2.imshow('prob', dt.prob)

        (x,y,w,h) = bbox
        result = frame.copy()
        cv2.rectangle(result, (x,y), (x+w,y+h), (0, 255, 255), 2)
        cv2.imshow(video_name, result)

        key = cv2.waitKey(300)
        if key == 27:
            break
        elif key == 112 or from_seq and not dt.inited:
            dt.init(frame, bbox)
        

        if from_seq:
            if idx >= len(seq):
                break
            else:
                frame = cv2.imread(seq[idx])
                idx += 1
        else: 
            success, frame = cap.read()
        
        t2 = time.clock()
        print "total speed: %ffps."% (1.0/(t2-t1))