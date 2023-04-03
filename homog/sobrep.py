import cv2
import numpy as np


image = cv2.imread('/home/marcelo/projetos/detectores/homography/imagens/2_red_h.jpg')

(B, G, R) = cv2.split(image)

zeros = np.zeros(image.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))

cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.imwrite('2-azul.jpg', cv2.merge([B, zeros, zeros]))

cv2.waitKey(0)

# from PIL import Image
#
# img1 = Image.open(r"1-verde.jpg")
# img2 = Image.open(r"2-vermelho.jpg")
# print(type(img1))
# img1.paste(img2, (0,0))
#
# img1.show()