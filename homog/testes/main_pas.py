import cv2 as cv
from homog import featuresutil as ft

img1 = '../../imagens/pas_eo/p4.jpg'
img2 = '../../imagens/pas_eo/1.jpg'


img_out, good, k1, k2 = ft.sift_correspondencias(img1, img2)

cv.imshow(f'{len(k1)}:{str(len(good))}', img_out)
#cv.imwrite(saida, img_out)
cv.waitKey(0)

