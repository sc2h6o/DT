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
video_name = "Shaking/img"
video_transpose = False
video_resize = (960, 540)
# bbox = 198,214,34,81
bbox = 225,135,61,71
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
class DeepTracker:
	def __init__(self):
		caffe.set_device(0)
		caffe.set_mode_gpu()
		self.inited = False
		self.prob = None
		self.bbox = (0,0,0,0)
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

	def makeLabels(self, bbox, box_large, w_sm, h_sm, range_out = 0.25, range_in = 0.1, scale=0.25):
		(_x,_y,_w,_h) = bbox
		(x,y,w,h) = box_large
		labels = np.zeros((1,h_sm,w_sm))
		rad_out = scale*range_out*40
		rad_in = scale*range_in*min(min(_w,_h),40)
		cx = scale*(_x+0.5*_w-x)
		cy = scale*(_y+0.5*_h-y)
		# x1 = scale*(_x-x)
		# y1 = scale*(_y-x)
		# x2 = scale*(_x+_w-x)
		# y2 = scale*(_y+_h-x)

		labels[0, rdint(cy-rad_out):rdint(cy+rad_out), rdint(cx-rad_out):rdint(cx+rad_out)] = -1
		labels[0, rdint(cy-rad_in):rdint(cy+rad_in), rdint(cx-rad_in):rdint(cx+rad_in)] = 1
		# print "*****************************"
		# print (w_sm,h_sm,rad_out,rad_in)
		return labels

	def makeLabels2(self, bbox, box_large, w_sm, h_sm, range_out = 0.25, range_in = 0.1, scale=0.25):
		(_x,_y,_w,_h) = bbox
		(x,y,w,h) = box_large
		labels = np.zeros((1,h_sm,w_sm))
		# rad_out = scale*range_out*40
		# rad_in = scale*range_in*min(min(_w,_h),40)
		# cx = scale*(_x+0.5*_w-x)
		# cy = scale*(_y+0.5*_h-y)
		pad_w_out = range_out*_w
		pad_h_out = range_out*_h
		pad_w_in = range_in*_w
		pad_h_in = range_in*_h
		labels[0, rdint(scale*(_y-y-pad_h_out)):rdint(scale*(_y-y+_h+pad_h_out)), rdint(scale*(_x-x-pad_w_out)):rdint(scale*(_x-x+_w+pad_w_out))] = -1
		labels[0, rdint(scale*(_y-y+pad_h_in)):rdint(scale*(_y-y+_h-pad_h_in)), rdint(scale*(_x-x+pad_w_in)):rdint(scale*(_x-x+_w-pad_w_in))] = 1
		# labels[0, rdint(cy-rad_out):rdint(cy+rad_out), rdint(cx-rad_out):rdint(cx+rad_out)] = -1
		# labels[0, rdint(cy-rad_in):rdint(cy+rad_in), rdint(cx-rad_in):rdint(cx+rad_in)] = 1
		return labels

	def getFeat(self, frame, box_large):
		(x,y,w,h) = box_large
		data = self.transpose(frame, box_large)
		self.featnet.blobs['data'].reshape(1,3,h,w)
		self.featnet.blobs['data'].data[0] = data
		self.featnet.forward()
		feat = self.featnet.blobs['feat'].data[0]
		return feat

	def update(self, frame, bbox ,step = 16):
		t1 = time.clock()
		(x,y,w,h) = box_large = padding(bbox, 1.0, 60)
		feat = self.getFeat(frame, box_large)
		(c_sm, h_sm, w_sm) = feat.shape
		labels = self.makeLabels(bbox, box_large, w_sm, h_sm)
		labels_seg = self.makeLabels2(bbox, box_large, w_sm, h_sm)

		self.solver.net.blobs['data'].reshape(1,c_sm,h_sm,w_sm)
		self.solver.net.blobs['data'].data[0] = feat
		self.solver.net.blobs['labels'].reshape(1,1,h_sm, w_sm)
		self.solver.net.blobs['labels'].data[0] = labels
		self.solver.net.blobs['labels_seg'].reshape(1,1,h_sm, w_sm)
		self.solver.net.blobs['labels_seg'].data[0] = labels_seg

		self.solver.step(step)
		t2 = time.clock()
		print 'update takes %f seconds.' % (1.0*(t2-t1))

	def init(self, frame, bbox):
		self.solver = caffe.SGDSolver(proto_solver)
		self.featnet = caffe.Net(proto_feat,model_feat,caffe.TEST)

		self.bbox = bbox
		# (_x,_y,_w,_h) = bbox
		# scale = target_size / max(_w,_h)
		# (_x,_y,_w,_h) = bbox = scaleBox(bbox, scale)
		frame = cv2.resize(frame,(0,0),fx=scale,fy=scale)
		self.update(frame, bbox, 2048)
		self.inited = True
		self.prob = np.zeros((frame.shape[0],frame.shape[1]))

	def UpdateSize(self, seg_big, bbox, box_large, smin = 0.8, smax = 1.2, step = 0.04, lamda = 0.4):
		(_x,_y,_w,_h) = bbox
		(x,y,w,h) = box_large
		cx = _x + 0.5*_w
		cy = _y + 0.5*_h
		w_new = h_new = 0
		score_size = -9e5
		s = smin
		while s < smax:
			hh = _h*s
			ww = _w*s
			score_temp = seg_big[rdint(cy-y-0.5*hh):rdint(cy-y+0.5*hh),rdint(cx-x-0.5*ww):rdint(cx-x+0.5*ww)].sum()  -  lamda*ww*hh
			if score_temp > score_size:
				score_size = score_temp
				w_new = ww
				h_new = hh
			s = s + step
		_x = rdint(cx-0.5*w_new)
		_y = rdint(cy-0.5*h_new)
		_w = rdint(w_new)
		_h = rdint(h_new)
		return (_x, _y, _w, _h)


	def track(self, frame):
		(_x,_y,_w,_h) = bbox =  self.bbox
		# scale = target_size / max(_w,_h)
		# (_x,_y,_w,_h) = bbox = scaleBox(bbox, scale)
		# frame = cv2.resize(frame,(0,0),fx=scale,fy=scale)

		(x,y,w,h) = box_large = padding(bbox, 0.7, 35)
		feat = self.getFeat(frame, box_large)
		(c_sm, h_sm, w_sm) = feat.shape
		self.solver.net.blobs['data'].reshape(1,c_sm,h_sm,w_sm)
		self.solver.net.blobs['labels'].reshape(1,1,h_sm, w_sm)
		self.solver.net.blobs['labels_seg'].reshape(1,1,h_sm, w_sm)
		self.solver.net.blobs['data'].data[0] = feat
		self.solver.net.forward()

		score = softmax(self.solver.net.blobs['score'].data[0])
		seg =  softmax(self.solver.net.blobs['score_seg'].data[0])
		# color = softmaxColor(self.solver.net.blobs['score'].data[0])
		# color = cv2.resize(color, (w_sm*4,h_sm*4))
		score_big = cv2.resize(score, (w_sm*4,h_sm*4))
		seg_big = cv2.resize(seg, (w_sm*4,h_sm*4))
		self.prob = score_big
		

		if score_big.max() > 0.00:
			cx = score_big.argmax() % (4*w_sm)
			cy = score_big.argmax() // (4*w_sm)
			_x = rdint(x + cx - 0.5*_w)
			_y = rdint(y + cy - 0.5*_h)
			bbox = (_x,_y,_w,_h)
			# bbox = self.UpdateSize(seg_big, bbox, box_large)
			self.bbox = bbox = scaleBox(bbox, 1/scale)
			self.update(frame, bbox)
		return self.bbox



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
			bbox = dt.track(frame)
			cv2.imshow('prob', dt.prob)

		(x,y,w,h) = bbox
		result = frame.copy()
		cv2.rectangle(result, (x,y), (x+w,y+h), (0, 255, 255), 2)
		cv2.imshow(video_name, result)

		key = cv2.waitKey(3)
		if key == 27:
			break
		elif key == 112 or from_seq and not dt.inited:
			cv2.waitKey(10)
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