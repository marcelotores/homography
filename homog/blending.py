import numpy as np
import cv2 as cv

img1 = cv.imread('1-verde.jpg')
img2 = cv.imread('2-vermelho.jpg')
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"
dst = cv.addWeighted(img1, 0.5, img2, 0.7, 0)


cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()