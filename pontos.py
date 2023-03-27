# python pontos.py imagem.jpg
import cv2
import sys

path = sys.argv[1]
image = cv2.imread(path)
center_coordinates = (164, 319,)
radius = 1
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Draw a circle with blue line borders of thickness of 2 px
image = cv2.circle(image, center_coordinates, radius, color, thickness)

cv2.imshow('Imagem', image)
cv2.waitKey(0)
