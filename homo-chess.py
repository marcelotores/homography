import cv2 as cv
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import cv2

im = cv.imread('1_re.jpg')
im2 = cv.imread('2_red.jpg')
#cv.imshow('Original', chess)
# cv.waitKey(0)

#source coordinates
# src = np.array([391, 100,
#                 14, 271,
#                 347, 624,
#                 747, 298,]).reshape((4, 2))

## imagem 1
src = np.array([365, 189,
                164, 321,
                609, 572,
                676, 235,]).reshape((4, 2))

src2 = np.array([208, 313,
                393, 558,
                691, 288,
                454, 222,]).reshape((4, 2))

# pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
# pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

#destination coordinates
# dst = np.array([100, 100,
#                 100, 650,
#                 650, 650,
#                 650, 100,]).reshape((4, 2))

## imagem1
dst = np.array([137, 264,
                99, 560,
                613, 570,
                587, 215,]).reshape((4, 2))

dst2 = np.array([194, 235,
                192, 534,
                696, 523,
                729, 209,]).reshape((4, 2))

#using skimage’s transform module where ‘projective’ is our desired parameter

tform = transform.estimate_transform('projective', src, dst)
tform2 = transform.estimate_transform('projective', src2, dst2)

h, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
h2, status2 = cv2.findHomography(src2, dst2)

tf_img = transform.warp(im, tform.inverse)
tf_img2 = transform.warp(im2, tform2.inverse)

#tf_img = transform.warp(chess, h)

#tf_img = cv2.warpPerspective(chess, np.array(tform), (chess.shape[1], chess.shape[0]))
teste = np.array(tform2)


#plotting the transformed image
fig, ax = plt.subplots()
#ax.imshow(tf_img)


#resized_image = cv2.resize(tf_img, (800, 598))

cv.imwrite('imagem.jpg', tf_img)

<<<<<<< HEAD
#cv.imshow('Imagem 1', tf_img)
cv.imshow('Imagem 2', tf_img2)
=======
cv.imshow('Imagem 1', tf_img)
#cv.imshow('Imagem 2', tf_img2)
>>>>>>> 01ef61c247b4915ff4b2f8a71f35b514b53aea08



#cv2.imwrite('output.jpg',dst)

#plt.show()

#im_v = cv2.vconcat([tf_img, tf_img])

#cv2.imshow('sea_image.jpg', im_v)
cv.imshow('Distorcida2', tf_img)

cv.waitKey(0)
_ = ax.set_title('projective transformation')