import cv2
import numpy as np
import featuresutil as ft

im_src = cv2.imread('../imagens/original1.jpg')

ponto_1 = [357, 227]
ponto_2 = [374, 351]
ponto_11 = [424, 228]
ponto_22 = [430, 323]

# ('357', '227'), ('374', '351'), ('424', '228'), ('430', '323')]
#distancia = ft.distancia_entre_pontos(ponto_1, ponto_2)
#print(distancia)

im_src = ft.reta(ponto_1, ponto_2, im_src)
im_src = ft.reta(ponto_11, ponto_22, im_src)
#im_src = ft.pontos(im_src, 208, 256)

#im_src = ft.equacao_reta(ponto_1, ponto_2, im_src)

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
#cv2.imshow("Nova imagem", nova_imagem)

cv2.imwrite('../imagens/imagem1_redimensionada.jpg', im_out)

cv2.waitKey(0)