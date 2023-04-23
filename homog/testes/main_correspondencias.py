import cv2 as cv
import numpy as np
import sys

from homog import featuresutil as ft

out, good = ft.sift_correspondencias('../imagens/motor1.jpg', '../imagens/motor2.jpg')
out2, good2 = ft.sift_correspondencias('../imagens/motor1.jpg', '../imagens/motor_deformado.jpg')


cv.imshow(f'sem homo: {str(len(good))}', out)

cv.imshow(f'com homo: {str(len(good2))}', out2)

cv.waitKey(0)
