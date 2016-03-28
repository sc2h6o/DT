import numpy as np

class DataBase:
    def __init__(self, net):
        self.idx = 0
        self.net = net

    def initData(self, batch_size, w, h, data):
        self.net.blobs['data'].reshape(1, 3, h, w)
        self.net.blobs['rois'] .reshape(batch_size, 5)
        self.net.blobs['labels'].reshape(batch_size)
        self.net.blobs['bbox_targets'].reshape(batch_size, 4)
        self.net.blobs['bbox_inside_weights'].reshape(batch_size, 4)
        self.net.blobs['bbox_outside_weights'].reshape(batch_size, 4)
        self.net.blobs['data'].data[0] = data



    def insertSample(self, label, roi, delta):
        self.net.blobs['rois'].data[self.idx] = roi
        self.net.blobs['labels'].data[self.idx] =  label
        self.net.blobs['bbox_targets'].data[self.idx] = delta
        self.net.blobs['bbox_inside_weights'].data[self.idx] = np.array([1,1,0,0])
        self.net.blobs['bbox_outside_weights'].data[self.idx] = np.array([1,1,1,1])
        self.idx += 1
        if self.idx >= self.net.blobs['data'].data.shape[0]:
            self.idx = 0
