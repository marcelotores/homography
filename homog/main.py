import cv2 as cv
import numpy as np
import featuresutil as ft

imagem_original = cv.imread('../imagens/original1.jpg')
imagem_redimensionada = cv.imread('../imagens/imagem1_redimensionada.jpg')
img2 = cv.imread('../imagens/1_red_h.jpg')

ponto_retangulo_1, ponto_retangulo_2 = [429, 488], [502, 297]

im_src = ft.retangulo(ponto_retangulo_1, ponto_retangulo_2, imagem_redimensionada, True)


cv.imshow('Com retangulo', im_src)
cv.imshow('Original', imagem_original)
cv.waitKey(0)
