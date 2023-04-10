import cv2
import numpy as np
import featuresutil as ft

im_src = cv2.imread('../imagens/original1.jpg')
imagem_redimensionada = cv2.imread('../imagens/imagem1_redimensionada.jpg')

#ponto_1 = [357, 227]
#ponto_2 = [378, 365]

ponto_1 = [357, 227]
ponto_2 = [375, 365]

ponto_11 = [424, 228]
ponto_22 = [430, 323]

terceito_motor_inicio, terceito_motor_fim = [195, 163], [250, 310]
quarto_motor_inicio, quarto_motor_fim = [224, 119], [274, 290]
# pontos do retângulo
ponto1_retangulo_1, ponto2_retangulo_1 = [429, 488], [502, 297]
ponto1_retangulo_2, ponto2_retangulo_2 = [223, 297], [291, 493]


# Original
# ponto_retangulo_1, ponto_retangulo_2 = [336, 369], [498, 274]


#im_src = ft.reta2(ponto_1, ponto_2, im_src)
#im_src = ft.reta2(ponto_11, ponto_22, im_src)
#im_src = ft.reta2(terceito_motor_inicio, terceito_motor_fim, im_src)
#im_src = ft.reta2(quarto_motor_inicio, quarto_motor_fim, im_src)
im_src = ft.circulo(ponto_2, 10, im_src)
im_src = ft.circulo(ponto_22, 10, im_src)
im_src = ft.circulo(terceito_motor_fim, 7, im_src)
im_src = ft.circulo(quarto_motor_fim, 7, im_src)

#final = ft.retangulo(ponto_retangulo_1, ponto_retangulo_2, imagem_redimensionada, True)

##im_src = ft.reta(ponto_1, ponto_2, im_src)
##im_src = ft.reta(ponto_11, ponto_22, im_src)



## imagem 1
pts_src = np.array([370, 203, 161, 318, 603, 563, 679, 252]).reshape((4, 2))

## imagem 1
pts_dst = np.array([152, 177, 152, 517, 699, 518, 699, 178]).reshape((4, 2))

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

final = ft.retangulo(ponto1_retangulo_1, ponto2_retangulo_1, imagem_redimensionada, True)
final = ft.retangulo2(ponto1_retangulo_2, ponto2_retangulo_2, imagem_redimensionada)
cv2.imshow('Original', im_src)
cv2.imshow("findHomo", final)
#cv2.imshow("Nova imagem", nova_imagem)

cv2.imwrite('../imagens/imagem1_redimensionada.jpg', im_out)

cv2.waitKey(0)