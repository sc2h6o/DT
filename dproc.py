import numpy as np
import cv2
from utils import *

def softmax(blob):
	zp = np.exp(blob[1])
	zn = np.exp(blob[0])
	return zp/(zp+zn)

def softmaxColor(blob):
	color = np.zeros((3, blob.shape[1], blob.shape[2]))
	zs = np.exp(blob).sum(axis=0)
	cmap = [[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0]]
	for i in range(1,6):
		z = np.exp(blob[i])/zs
		color[0] = color[0] + cmap[i-1][0] * z
		color[1] = color[1] + cmap[i-1][1] * z
		color[2] = color[2] + cmap[i-1][2] * z
		return color.transpose(1,2,0)

def makeLabels(bbox, box_large, w_sm, h_sm, range_out = 0.25, range_in = 0.1, scale=0.25):
	(_x,_y,_w,_h) = bbox
	(x,y,w,h) = box_large
	labels = np.zeros((1,h_sm,w_sm))
	rad_out = scale*range_out*40
	rad_in = scale*range_in*min(min(_w,_h),40)
	cx = scale*(_x+0.5*_w-x)
	cy = scale*(_y+0.5*_h-y)

	labels[0, rdint(cy-rad_out):rdint(cy+rad_out), rdint(cx-rad_out):rdint(cx+rad_out)] = -1
	labels[0, rdint(cy-rad_in):rdint(cy+rad_in), rdint(cx-rad_in):rdint(cx+rad_in)] = 1
	return labels

def makeLabelsSeg(bbox, box_large, w_sm, h_sm, range_out = 0, range_in = 0, scale=0.25):
	(_x,_y,_w,_h) = bbox
	(x,y,w,h) = box_large
	labels = np.zeros((1,h_sm,w_sm))
	pad_w_out = range_out*_w
	pad_h_out = range_out*_h
	pad_w_in = range_in*_w
	pad_h_in = range_in*_h

	labels[0, rdint(scale*(_y-y-pad_h_out)):rdint(scale*(_y-y+_h+pad_h_out)), rdint(scale*(_x-x-pad_w_out)):rdint(scale*(_x-x+_w+pad_w_out))] = -1
	labels[0, rdint(scale*(_y-y+pad_h_in)):rdint(scale*(_y-y+_h-pad_h_in)), rdint(scale*(_x-x+pad_w_in)):rdint(scale*(_x-x+_w-pad_w_in))] = 1
	return labels

def transpose(frame, mean, bbox=None):
	if bbox == None:
		data = frame
	else:
		(x,y,w,h) = bbox
		pad = 200
		_frame = np.zeros((frame.shape[0] + 2*pad, frame.shape[1] + 2*pad, 3))
		_frame[pad:pad+frame.shape[0], pad:pad+frame.shape[1], :] = frame
		data = _frame[pad+y:pad+y+h, pad+x:pad+x+w, :]
		# data = cv2.resize(data, (scale*w,scale*h))
	data = data - mean
	data = data.transpose((2,0,1))
	return data