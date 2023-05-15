import cv2 as cv
import numpy as np

from homog import featuresutil as ft

img1 = '../../imagens/pas_eo/Descarga14.JPG'

img_sem_filtro = cv.imread(img1)

# Kernel
kernel = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
#img_com_filtro = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

# Aplicação do filtro blur do opencv a imagem
img_com_filtro = cv.blur(img_sem_filtro, ksize=(3, 3))

# Colocando a imagem na escala de cinza
gray = cv.cvtColor(img_com_filtro, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img_sem_filtro, cv.COLOR_BGR2GRAY)

# Definindo o objeto sift
sift = cv.SIFT_create()

# Calculando os detectores e descritores
kp, des = sift.detectAndCompute(gray, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Usando a função drawKeypoints para desenhar os pontos na imagem filtro.
img = cv.drawKeypoints(gray, kp, img_com_filtro)
img2 = cv.drawKeypoints(gray2, kp2, img_sem_filtro)
# , flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

#cv.imshow(str(len(kp)), img_com_filtro)

# Redimensionado o tamanho da imagem
imgS = ft.res_img(img, 800)
img2S = ft.res_img(img2, 800)
#img_com_filtroS = ft.res_img(img_com_filtro, 800)


cv.imshow(f'Imagem com Filtro: {str(len(kp))}', imgS)
cv.imshow(f'Imagem sem Filtro: {str(len(kp2))}', img2S)
cv.waitKey(0)
#cv.imwrite('sift_keypoints.jpg',img)