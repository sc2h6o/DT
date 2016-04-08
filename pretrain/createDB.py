import cv2
import os
import math
import numpy as np
import random
import h5py

sequences = ['Basketball', 'Bird1', 'BlurCar1', 'Bolt2', 'Box', 'Car1', 'CarDark',\
 'ClifBar', 'Diving', 'DragonBaby', 'FaceOcc1', 'Freeman1', 'Freeman4', 'Girl', 'Girl2', 'Human3', 'Human6',\
  'KiteSurf', 'Liquor', 'Ironman', 'Skating1', 'Soccer', 'Tiger1', 'Woman']

dirIn = '/media/syc/My Passport/_dataset/tracking2013/'
dirOut = '../data/'
dirTrain = "train.txt"
dirTest =  "test.txt"

testNum = 2000
zeroNum = 4

samples = []

for seq in sequences:
    dirImg = dirIn + seq + '/img/'
    dirGt = dirIn + seq + '/groundtruth_rect.txt'
    if not os.path.exists(dirImg[:-1]):
        print 'dirImg not exist:%s' % dirImg
    if not os.path.exists(dirOut[:-1]):
        print 'makedir dirOut: %s' % dirOut
        os.mkdir(dirOut)
    print 'making label dataset for: %s' % seq


    gt = open(dirGt)
    images = os.listdir(dirImg)
    images.sort()
    for i,line in enumerate(gt):
        line = line.replace('\t', ',')
        line = dirImg + images[i] + '\t' + line
        samples.append(line)

print "shuffling samples"
random.shuffle(samples)

train  = open(dirOut+dirTrain, 'wb')
test = open(dirOut+dirTest, 'wb')
for i in range(len(samples)):
    if i < len(samples) - testNum:
        train.write(samples[i])
    else:
        test.write(samples[i])
train.close()
test.close()