import cv2
import numpy as np
import featuresutil as ft

im_src = cv2.imread('../imagens/1_red.jpg')
im_src = ft.pontos(im_src, 365, 189, 'azul')
im_src = ft.pontos(im_src, 164, 321, 'azul')
im_src = ft.pontos(im_src, 609, 572, 'azul')
im_src = ft.pontos(im_src, 676, 235, 'azul')

im_src = ft.pontos(im_src, 137, 264, 'verde')
im_src = ft.pontos(im_src, 137, 570, 'verde')
im_src = ft.pontos(im_src, 613, 570, 'verde')
im_src = ft.pontos(im_src, 613, 264, 'verde')

## imagem 1
pts_src = np.array([365, 189,
                 164, 321,
                 609, 572,
                 676, 235,]).reshape((4, 2))

## imagem 1
pts_dst = np.array([137, 264,
                 137, 570,
                 613, 570,
                 613, 264,]).reshape((4, 2))

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

cv2.imshow('Original', im_src)
cv2.imshow("findHomo", im_out)

#cv2.imwrite('../imagens/voltando.jpg', im_out)

cv2.waitKey(0)