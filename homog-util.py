import cv2 as cv
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import sys

# Passa imagem e pontos.
#img1 = sys.argv[1]
im = cv.imread('1_re.jpg')


src = np.array([365, 189,
                164, 321,
                609, 572,
                676, 235]).reshape((4, 2))

dst = np.array([137, 264,
                99, 560,
                613, 570,
                587, 215]).reshape((4, 2))


# using skimage’s transform module where ‘projective’ is our desired parameter
tform = transform.estimate_transform('projective', src, dst)

# h, status = cv.findHomography(src, dst)
tf_img = transform.warp(im, tform.inverse)

fig, ax = plt.subplots()

cv.imshow('transform.estimate_transform', tf_img)

cv.waitKey(0)
#_ = ax.set_title('projective transformation')
