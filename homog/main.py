import cv2 as cv
import numpy as np
import featuresutil as ft

img1 = cv.imread('../imagens/1_red.jpg')
img2 = cv.imread('../imagens/1_red_h.jpg')

#ft.pontos(img1, 353, 207)

kp1, kp2, good = ft.sift_correspondencias(img1, img2)


img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# print('Quantidade de Correspondências: ', len(good))


cv.imshow('Correspondencias', img3)
cv.waitKey(0)
