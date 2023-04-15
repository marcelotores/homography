import cv2 as cv
import numpy as np
import sys

from homog import featuresutil as ft

#img1 = sys.argv[1]
#img2 = sys.argv[2]

img1 = cv.imread('../imagens/155850.jpg')
img2 = cv.imread('../imagens/155853.jpg')
#img2 = cv.imread('im_deformada.jpg')



kp1, kp2, good = ft.sift_correspondencias(img1, img2)

img1 = cv.imread('../imagens/155850.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('../imagens/155853.jpg', cv.IMREAD_GRAYSCALE)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow(str(len(good)), img3)
cv.imwrite('sem_homografia.jpg', img3)
cv.waitKey(0)
