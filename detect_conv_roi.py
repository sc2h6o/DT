import cv2
import os
import sys
sys.path.append("/home/syc/py-faster-rcnn/caffe-fast-rcnn/python")
import caffe
import numpy as np
import random
import time
from math import *
from utils import *
from DataBase import DataBase

video_dir = '/media/syc/My Passport/_dataset/tracking2013/'
video_name = "Liquor/img"
video_transpose = False
video_resize = (960, 540)
bbox = 256,152,73,210
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

target_size = 127.0
pad_in = 32

class DeepTracker:
	def __init__(self):
		caffe.set_device(0)
		caffe.set_mode_gpu()
		self.inited = False
		self.prob = None
		self.pad_w = self.pad_h = 0
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

	def makeLabels(self, bbox, box_large, w_sm, h_sm, range_out = 12, range_in = 6, scale=0.25):
		(_x,_y,_w,_h) =  bbox
		(x,y,w,h) = box_large
		labels = np.zeros((1,h_sm,w_sm))
		rad_out = scale*range_out
		rad_in = scale*range_in
		cx = scale*(_x-x+self.pad_w)
		cy = scale*(_y-y+self.pad_h)
		labels[0, rdint(cy-rad_out):rdint(cy+rad_out), rdint(cx-rad_out):rdint(cx+rad_out)] = -1
		labels[0, rdint(cy-rad_in):rdint(cy+rad_in), rdint(cx-rad_in):rdint(cx+rad_in)] = 1
		return labels

	def getFeat(self, frame, bbox, box_large):
		(_x,_y,_w,_h) = bbox
		(x,y,w,h) = box_large
		w_feat = int_(0.25*w) - rdint(0.25*(_w-2*self.pad_w)) 
		h_feat = int_(0.25*h) - rdint(0.25*(_h-2*self.pad_h)) 
		data = self.transpose(frame, box_large)
		self.featnet.blobs['data'].reshape(1,3,h,w)
		self.featnet.blobs['data'].data[0] = data
		self.featnet.blobs['rois'].reshape(w_feat*h_feat,5)
		for i in range(w_feat):
			for j in range(h_feat):
				idx = j + i*h_feat
				self.featnet.blobs['rois'].data[idx] = np.array([0,4*i,4*j,4*i+_w-2*self.pad_w,4*j+_h-2*self.pad_h])
		self.featnet.forward()
		pool = self.featnet.blobs['roi_pool_conv5'].data
		feat = pool.reshape(w_feat,h_feat,1024).transpose((2,1,0))
		return feat

	def update(self, frame, bbox ,step = 16):
		t1 = time.clock()
		(_x,_y,_w,_h) = bbox
		self.pad_w = min(_w//2, pad_in)
		self.pad_h = min(_h//2, pad_in)

		(x,y,w,h) = box_large = padding(bbox, 1.0, 60)
		feat = self.getFeat(frame, bbox, box_large)
		(c_sm, h_sm, w_sm) = feat.shape
		labels = self.makeLabels(bbox,box_large,w_sm, h_sm)

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
		(x,y,w,h) = box_large = padding(bbox, 0.7, 35)
		feat = self.getFeat(frame, bbox, box_large)
		(c_sm, h_sm, w_sm) = feat.shape
		self.solver.net.blobs['data'].reshape(1,c_sm,h_sm,w_sm)
		self.solver.net.blobs['labels'].reshape(1,1,h_sm, w_sm)
		self.solver.net.blobs['data'].data[0] = feat
		self.solver.net.forward()

		score = softmax(self.solver.net.blobs['score'].data[0])
		score_big = cv2.resize(score, (4*w_sm,4*h_sm))
		self.prob = score_big.copy() ##
		dx = score_big.argmax() % (4*w_sm)
		dy = score_big.argmax() // (4*w_sm)
		_x = x+dx-self.pad_w
		_y = y+dy-self.pad_h 
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