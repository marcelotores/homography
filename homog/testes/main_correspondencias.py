import cv2 as cv
import numpy as np
import sys

from homog import featuresutil as ft

#img1 = sys.argv[1]
#img2 = sys.argv[2]

#img1 = cv.imread('../imagens/copo1.jpg')
#img2 = cv.imread('../imagens/copo2.jpg')
#img2 = cv.imread('moc_deformado.jpg')

out, good = ft.sift_correspondencias('../imagens/moc1.jpg', '../imagens/moc2.jpg')

#kp1, kp2, good = ft.sift_correspondencias(img1, img2)
#kp1, kp2, good = ft.orb_correspondencias(img1, img2)


#img1 = cv.imread('../imagens/c1.jpg', cv.IMREAD_GRAYSCALE)
#img2 = cv.imread('../imagens/c2.jpg', cv.IMREAD_GRAYSCALE)

#img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow(str(len(good)), out)
#cv.imshow(str(len(good)), cv.resize(img3, (660, 540)))
#cv.imshow(str(len(good)), img3)
#cv.imwrite('cola_sem_homografia.jpg', img3)
cv.waitKey(0)
