import cv2 as cv
import numpy as np

from homog import featuresutil as ft

img1 = '../../imagens/pas_eo/p4.jpg'

img = cv.imread(img1)


kernel = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
#img_com_filtro = cv.filter2D(src=img, ddepth=-1, kernel=kernel)
img_com_filtro = cv.blur(img, ksize=(3, 3))

gray = cv.cvtColor(img_com_filtro, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
img = cv.drawKeypoints(gray, kp, img_com_filtro)
# , flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

cv.imshow(str(len(kp)), img_com_filtro)
cv.waitKey(0)
#cv.imwrite('sift_keypoints.jpg',img)