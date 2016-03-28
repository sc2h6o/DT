import numpy as np
import cv2
import random


def int_(x):
	if x==int(x):
		return int(x)
	else:
		return int(x+1)

def IoU(box1, box2):
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

def transform(bbox, delta=(0,0,-0,-0)):
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

def transform_inv(bbox, delta):
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

def padding(bbox, scale):
	x,y,w,h = bbox
	pad = int(min(w,h))
	x -= pad
	y -= pad
	w += 2 * pad
	h += 2 * pad
	return (x,y,w,h)

def rdint(x):
	return int(round(x))

def softmax(blob):
	zp = np.exp(blob[1])
	zn = np.exp(blob[0])
	return zp/(zp+zn)