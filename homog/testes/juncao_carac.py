import cv2 as cv
import numpy as np
from homog import featuresutil as ft

path1 = '../../imagens/pas_eo/p4.jpg'
path2 = '../../imagens/pas_eo/1.jpg'

img_patch1 = cv.imread(path1)
img_original = cv.imread(path2)

kp1, des1 = ft.sift_detectores_e_descritores(img_patch1)
kp2, des2 = ft.sift_detectores_e_descritores(img_original)

img_patch1_out = cv.drawKeypoints(img_patch1, kp1, img_patch1)
cv.imshow(f'Qtq keypoints: {len(kp1)}', img_patch1_out)

# N principais características
# n = 10
# kp1 = sorted(kp1, key=lambda x: -x.response)[:n]

# Características
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img_patch1, kp1, img_original, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#cv.imshow('', img3)
cv.waitKey(0)

print(len(kp1))
print(len(kp2))