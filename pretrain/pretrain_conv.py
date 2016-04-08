import cv2
import os
import sys
sys.path.append("/home/syc/py-faster-rcnn/caffe-fast-rcnn/python")
sys.path.append("../")
import caffe
import numpy as np
import random
import time
import dproc
from utils import *


model_dir = '../model/'
data_dir = '../data/'
file_train = data_dir + "train.txt"
file_test = data_dir + "test.txt"
solver = model_dir + 'solver_pretrain.prototxt'
model =  model_dir + "ZF_faster_rcnn_final.caffemodel"
model_out = model_dir + "ZF_conv.caffemodel"

train_iter = 100


class DeepTrainer:
	def __init__(self):
		caffe.set_device(0)
		caffe.set_mode_gpu()
		self.mean = np.array([102.9801, 115.9465, 122.7717])
		self.t1 = self.t2 = 0

	def update(self, frame, bbox ,scale = 0.25, step = 1):
		(x,y,w,h) = box_large = padding(bbox, 1.0, 45)
		data = dproc.transpose(frame, self.mean, box_large)
		w_sm = int_(scale * data.shape[2])
		h_sm = int_(scale * data.shape[1])
		labels = dproc.makeLabels(bbox, box_large, w_sm, h_sm)
		labels_seg = dproc.makeLabelsSeg(bbox, box_large, w_sm, h_sm)

		self.solver.net.blobs['data'].reshape(1,3,h,w)
		self.solver.net.blobs['data'].data[0] = data
		self.solver.net.blobs['labels'].reshape(1,1,h_sm, w_sm)
		self.solver.net.blobs['labels'].data[0] = labels
		self.solver.net.blobs['labels_seg'].reshape(1,1,h_sm, w_sm)
		self.solver.net.blobs['labels_seg'].data[0] = labels_seg

		# calculate the time consumed
		self.t1 = time.clock()
		print 'prepare uses %f s' % (self.t1 - self.t2)
		self.solver.step(step)
		self.t2 = time.clock()
		print 'update uses %f s' % (self.t2 - self.t1)


	def parseLine(self, line):
		pair = line[:-1].split('\t')
		file_img= pair[0]
		bbox = pair[1].split(',')
		bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
		return file_img, bbox


	def train(self, solver, file_train, file_test, model = None):
		self.solver = caffe.SGDSolver(solver)
		if model:
			self.solver.net.copy_from(model)

		test = open(file_test)

		for i in range(train_iter):
			print "-----------------------------------------"
			print "training iteration %d" % i
			print "-----------------------------------------"
			train = open(file_train)
			for line in train:
				file_img, bbox = self.parseLine(line)
				frame = cv2.imread(file_img)
				self.update(frame, bbox)
			train.close()
			self.solver.net.save(model_out)


if __name__ == "__main__":
	
	dt = DeepTrainer()
	dt.train(solver, file_train, file_test, model)