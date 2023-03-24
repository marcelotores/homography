import cv2 as cv
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import cv2


chess = cv.imread('chess.png')
#cv.imshow('Original', chess)
# cv.waitKey(0)

#source coordinates
src = np.array([391, 100,
                14, 271,
                347, 624,
                747, 298,]).reshape((4, 2))

# pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
# pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

#destination coordinates
dst = np.array([100, 100,
                100, 650,
                650, 650,
                650, 100,]).reshape((4, 2))

#using skimage’s transform module where ‘projective’ is our desired parameter
tform = transform.estimate_transform('projective', src, dst)

h, status = cv2.findHomography(src, dst)

tf_img = transform.warp(chess, tform.inverse)
#tf_img = transform.warp(chess, h)

#tf_img = cv2.warpPerspective(chess, np.array(tform), (chess.shape[1], chess.shape[0]))
teste = np.array(tform)

print(type(teste))
print(type(tform))
print(type(h))
#plotting the transformed image
fig, ax = plt.subplots()
#ax.imshow(tf_img)
cv.imshow('Distorcida', tf_img)
cv.waitKey(0)
_ = ax.set_title('projective transformation')