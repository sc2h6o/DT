# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import yaml
from multiprocessing import Process, Queue

class DataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""


    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)


        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(1, 640,56,56)
        self._name_to_top_map['data'] = idx
        idx += 1

        # labels blob: holds a batch of N matrix, one-forth size of the image
        top[idx].reshape(1, 1,56,56)
        self._name_to_top_map['labels'] = idx
        idx += 1

        # labels blob: holds a batch of N matrix, one-forth size of the image
        top[idx].reshape(1, 1,56,56)
        self._name_to_top_map['labels_seg'] = idx
        idx += 1

        print 'DataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass 
