import cv2
import numpy as np
import featuresutil as ft

im_src = cv2.imread('../imagens/original1.jpg')
im_src = ft.pontos(im_src, 370, 203, 'B')
im_src = ft.pontos(im_src, 161, 318, 'B')
im_src = ft.pontos(im_src, 603, 563, 'B')
im_src = ft.pontos(im_src, 679, 252, 'B')
# motor
im_src = ft.pontos(im_src, 356, 224, 'R')
im_src = ft.pontos(im_src, 356, 223, 'R')
im_src = ft.pontos(im_src, 356, 220, 'R')
im_src = ft.pontos(im_src, 356, 219, 'R')
im_src = ft.pontos(im_src, 356, 217, 'R')
im_src = ft.pontos(im_src, 356, 216, 'R')
im_src = ft.pontos(im_src, 356, 215, 'R')
im_src = ft.pontos(im_src, 356, 212, 'R')





## imagem 1
pts_src = np.array([370, 203, 161, 318, 603, 563, 679, 252]).reshape((4, 2))

## imagem 1
pts_dst = np.array([152, 177, 152, 517, 699, 518, 699, 178]).reshape((4, 2))

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

cv2.imshow('Original', im_src)
cv2.imshow("findHomo", im_out)

#cv2.imwrite('../imagens/voltando.jpg', im_out)

cv2.waitKey(0)