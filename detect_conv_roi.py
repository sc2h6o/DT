import cv2
import os
import sys
sys.path.append("/home/syc/caffe-new/python")
import caffe
import numpy as np
import random
import time
from math import *
from utils import *
from DataBase import DataBase

video_dir = "../../dataset/"
video_name = "tiger2"
video_transpose = False
video_resize = (960, 540)
bbox = 32,60,68,78
# bbox = 100,20,60,60
(x,y,w,h) = bbox

batch_size = 4

from_seq = True

scale = 1

model_dir = 'model/'
data_dir = 'data/'
proto_solver = model_dir + 'solver.prototxt'
proto_feat = model_dir + 'feat.prototxt'
model_feat =  model_dir + "ZF_faster_rcnn_final.caffemodel"
# mean_file = model_dir + 'ilsvrc_2012_mean.npy'

target_size = 224

class DeepTracker:
    def __init__(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.inited = False
        self.prob = None
        self.mean = np.array([102.9801, 115.9465, 122.7717])
        self.idx = 0

    def transpose(self, frame, bbox=None):
        if bbox == None:
            data = frame
        else:
            (x,y,w,h) = bbox
            pad = 200
            _frame = np.zeros((frame.shape[0] + 2*pad, frame.shape[1] + 2*pad, 3))
            _frame[pad:pad+frame.shape[0], pad:pad+frame.shape[1], :] = frame
            data = _frame[pad+y:pad+y+h, pad+x:pad+x+w, :]
            # data = cv2.resize(data, (scale*w,scale*h))
        data = data - self.mean
        data = data.transpose((2,0,1))
        return data


    def saveImage(self, frame, bbox, idx):
        x,y,w,h = bbox
        cv2.imwrite('data/%d.jpg'%idx, frame[y:y+h,x:x+w,:])

    def makeLabels(self, w_sm, h_sm, scale_out = 0.12, scale_in = 0.06):
        labels = np.zeros((1,h_sm,w_sm))
        rad_out = scale_out*min(w_sm,h_sm)
        rad_in = scale_in*min(w_sm,h_sm)
        cx = 0.5*w_sm
        cy = 0.5*h_sm
        labels[0, rdint(cy-rad_out):rdint(cy+rad_out), rdint(cx-rad_out):rdint(cx+rad_out)] = -1
        labels[0, rdint(cy-rad_in):rdint(cy+rad_in), rdint(cx-rad_in):rdint(cx+rad_in)] = 1
        print "*****************************"
        print (w_sm,h_sm,rad_out,rad_in)
        return labels

    def getFeat(self, frame, bbox, box_large):
        (_x,_y,_w,_h) = bbox
        (x,y,w,h) = box_large
        w_feat = int_(0.25*w) - rdint(0.25*_w) + 1
        h_feat = int_(0.25*h) - rdint(0.25*_h) + 1
        data = self.transpose(frame, box_large)
        self.featnet.blobs['data'].reshape(1,3,h,w)
        self.featnet.blobs['data'].data[0] = data
        self.featnet.blobs['rois'].reshape(w_feat*h_feat,5)
        for i in range(w_feat):
            for j in range(h_feat):
                idx = j + i*w_feat
                self.featnet.blobs['rois'].data[idx] = np.array([0,4*i,4*j,4*i+_w,4*j+_h])
        self.featnet.forward()
        pool = self.featnet.blobs['roi_pool_conv5'].data
        c_pool = pool.shape[1]*pool.shape[2]*pool.shape[3]
        feat = np.zeros((c_pool,h_feat,w_feat))
        for i in range(w_feat):
            for j in range(w_feat):
                idx = j + i*w_feat
                feat[:,j,i] = pool[idx].flatten()
        return feat

    def update(self, frame, bbox ,step = 32):
        t1 = time.clock()
        (x,y,w,h) = box_large = padding(bbox, 0.6)
        feat = self.getFeat(frame, bbox, box_large)
        (c_sm, h_sm, w_sm) = feat.shape
        labels = self.makeLabels(w_sm, h_sm)

        self.solver.net.blobs['data'].reshape(1,c_sm,h_sm,w_sm)
        self.solver.net.blobs['data'].data[0] = feat
        self.solver.net.blobs['labels'].reshape(1,1,h_sm,w_sm)
        self.solver.net.blobs['labels'].data[0] = labels
        self.solver.step(step)
        t2 = time.clock()
        print 'update takes %f seconds.' % (1.0*(t2-t1))

    def init(self, frame, bbox):
        self.solver = caffe.SGDSolver(proto_solver)
        self.featnet = caffe.Net(proto_feat,model_feat,caffe.TEST)

        self.update(frame, bbox, 1024)
        
        self.inited = True
        self.prob = np.zeros((frame.shape[0],frame.shape[1]))

    def track(self, frame, bbox):
        (_x,_y,_w,_h) = bbox
        (x,y,w,h) = box_large = padding(bbox, 0.6)
        feat = self.getFeat(frame, bbox, box_large)
        (c_sm, h_sm, w_sm) = feat.shape
        self.solver.net.blobs['data'].reshape(1,c_sm,h_sm,w_sm)
        self.solver.net.blobs['data'].data[0] = feat
        self.solver.net.forward()

        ww = w - _w
        hh = h - _h
        score = softmax(self.solver.net.blobs['score'].data[0])
        score_big = cv2.resize(score, (ww,hh))
        self.prob = score_big.copy() ##
        cx = score_big.argmax() % ww
        cy = score_big.argmax() // hh
        dx = rdint(cx - 0.5*ww)
        dy = rdint(cy - 0.5*hh)
        _x += dx
        _y += dy
        bbox = (_x,_y,_w,_h)

        self.update(frame, bbox)
        return bbox



if __name__ == "__main__":
    
    dt = DeepTracker()

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
            cv2.imshow('prob', dt.prob)

        (x,y,w,h) = bbox
        result = frame.copy()
        cv2.rectangle(result, (x,y), (x+w,y+h), (0, 255, 255), 2)
        cv2.imshow(video_name, result)

        key = cv2.waitKey(3)
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