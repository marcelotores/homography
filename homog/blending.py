import numpy as np
import cv2 as cv

img1 = cv.imread('/home/infra/PycharmProjects/homography/homog/1-vermelho.jpg')
img2 = cv.imread('/home/infra/PycharmProjects/homography/homog/2-verde.jpg')
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"
dst = cv.addWeighted(img1, 0.7, img2, 0.2, 0)


cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()