import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average

io = cv2.imread('breakout_01.png')
cv2.imshow('color image', io)
# 208, 162, 3
print(f'shape: {io.shape}, type: {type(io)}')
#cv2.waitKey(0)

ig = cv2.cvtColor(io, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image', ig)
cv2.waitKey(0)

ig = cv2.resize(ig, (128, 128))
# average pooling def average_pooling(img, G=8): out = img.copy() H, W, C = img.shape Nh = int(H / G) Nw = int(W / G) for y in range(Nh): for x in range(Nw): for c in range(C): out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int) return out

def average_pooling(img, G=8):
    H, W = img.shape
    Nh = int(H/G)
    Nw = int(W/G)
    out = np.zeros((Nh, Nw))

    for y in range(0, Nh):
        for x in range(0, Nw):
                out[y, x] = np.mean(img[G*y:G*(y+1), G*x:G*(x+1)])
    return out

pg = average_pooling(ig)
cv2.imshow('ava pooling', pg)
cv2.waitKey(0)

def max_pooling(img, G=8):
    H, W = img.shape
    Nh = int(H/G)
    Nw = int(W/G)
    out = np.zeros((Nh, Nw))

    for y in range(0, Nh):
        for x in range(0, Nw):
                out[y, x] = np.max(img[G*y:G*(y+1), G*x:G*(x+1)])
    return out

pg = max_pooling(ig)
cv2.imshow('max pooling', pg)
cv2.waitKey(0)
