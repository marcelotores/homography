import cv2 as cv
import numpy as np
from homog import featuresutil as ft

path1 = '../../imagens/pas_eo/p4.jpg'
path2 = '../../imagens/pas_eo/1.jpg'


def imprime_informacoes(kp):
  if len(kp) == 0:
    print('Não há pontos')
    return
  print('Quantidade de Pontos: ', len(kp))
  print('Coordenadas')
  for i in range(len(kp)):
    print(kp[i], ' - ', kp[i].pt)

  print('Relevância')
  for i in range(len(kp)):
    print(kp[i], ' - ', kp[i].response)

img_patch1 = cv.imread(path1)
img_original = cv.imread(path2)

# Filtros
# blur = cv.blur(img,(3,3))
# GaussianBlur = cv.GaussianBlur(img,(5,5), 0)

# Filtro Mediana
#medianBlur_img_patch1 = cv.medianBlur(img_patch1, 3)
#medianBlur_img_original = cv.medianBlur(img_original, 3)

# Filtro Média

#medianBlur_img_patch1 = cv.blur(img_patch1,(3,3))
#medianBlur_img_original = cv.blur(img_original,(3,3))

# Filtro Gaussiano

# medianBlur_img_patch1 = cv.GaussianBlur(img_patch1, (3, 3), 0)
# medianBlur_img_original = cv.GaussianBlur(img_original, (3, 3), 0)

# Filtro Bilateral

medianBlur_img_patch1 = cv.bilateralFilter(img_patch1, 9,75,75)
medianBlur_img_original = cv.bilateralFilter(img_original, 9,75,75)

# medianBlur_original = cv.medianBlur(imagem_original,5)
# bilateralFilter = cv.bilateralFilter(img,9,75,75)
# cv2_imshow(blur)
# cv2_imshow(GaussianBlur)
# cv2_imshow(medianBlur)
# cv2_imshow(bilateralFilter)

# Detectores e descritores das duas imagens originais
kp1, des1 = ft.sift_detectores_e_descritores(img_patch1)
kp2, des2 = ft.sift_detectores_e_descritores(img_original)

# Detectores e descritores das duas imagens com filtros
kp1_medianBlur, des1_medianBlur = ft.sift_detectores_e_descritores(medianBlur_img_patch1)
kp2_medianBlur, des2_medianBlur = ft.sift_detectores_e_descritores(medianBlur_img_original)

# Desenhando os pontos sobre as imagens originais
img_patch1_out = cv.drawKeypoints(img_patch1, kp1, img_patch1)
original_out = cv.drawKeypoints(img_original, kp2, img_original)

# Desenhando os pontos sobre as imagens com filtros
medianBlur_img_patch1_out = cv.drawKeypoints(medianBlur_img_patch1, kp1_medianBlur, medianBlur_img_patch1)
medianBlur_img_original_out = cv.drawKeypoints(medianBlur_img_original, kp2_medianBlur, medianBlur_img_original)

print('kp1')
imprime_informacoes(kp1)

# print('kp1_medianBlur')
# imprime_informacoes(kp1_medianBlur)

print('Original')
imprime_informacoes(kp2)

#cv.imshow(f'Qtq keypoints (img_patch1): {len(kp1)}', img_patch1_out)
#cv.imshow(f'Qtq keypoints (img_original): {len(kp2)}', original_out)
#cv.imshow(f'Qtq keypoints (medianBlur_pat1): {len(kp1_medianBlur)}', medianBlur_img_patch1_out)
#cv.imshow(f'Qtq keypoints (medianBlur_ori): {len(kp2_medianBlur)}', medianBlur_img_original_out)


# N principais características
# n = 10
# kp1 = sorted(kp1, key=lambda x: -x.response)[:n]

# Características
bf = cv.BFMatcher()
# Correspondências imagens sem filtro
#matches = bf.knnMatch(des1, des2, k=2)

# Correspondências imagens com filtro
matches = bf.knnMatch(des1_medianBlur, des2_medianBlur, k=2)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.

# Desenhando imagem sem filtro
#img3 = cv.drawMatchesKnn(img_patch1, kp1, img_original, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Desenhando imagem com filtro
img3 = cv.drawMatchesKnn(medianBlur_img_patch1, kp1_medianBlur, medianBlur_img_original, kp2_medianBlur, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('', img3)
cv.waitKey(0)

print(len(kp1))
print(len(kp2))