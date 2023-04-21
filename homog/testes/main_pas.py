import cv2 as cv
from homog import featuresutil as ft

img1 = '../../imagens/pas_eo/p1.jpg'
img2 = '../../imagens/pas_eo/1.jpg'

img_out, good = ft.sift_correspondencias(img1, img2)
#img_out, good = ft.orb_correspondencias(img1, img2)

print(img_out.shape)

#nova = resized = cv.resize(img_out, (500, 500), interpolation = cv.INTER_AREA)
#nova = ft.res_img(img_out, 0)

cv.imshow(str(len(good)), img_out)
cv.imwrite('p6-p6.jpg', img_out)
cv.waitKey(0)