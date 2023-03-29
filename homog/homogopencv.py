import cv2
import numpy as np

im_src = cv2.imread('../imagens/2_red.jpg')

## imagem 2
pts_src = np.array([208, 313,
                393, 558,
                691, 288,
                454, 222,]).reshape((4, 2))

## imagem 2
pts_dst = np.array([194, 235,
                192, 534,
                696, 523,
                729, 209,]).reshape((4, 2))

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

cv2.imshow("findHomo", im_out)

cv2.imwrite('../imagens/2_red_h.jpg', im_out)

cv2.waitKey(0)