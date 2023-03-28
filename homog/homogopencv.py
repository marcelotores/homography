import cv2
import numpy as np

im_src = cv2.imread('../imagens/1_red.jpg')

pts_src = np.array([365, 189,
                    164, 321,
                    609, 572,
                    676, 235]).reshape((4, 2))

pts_dst = np.array([137, 264,
                    99, 560,
                    613, 570,
                    587, 215]).reshape((4, 2))

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

cv2.imshow("findHomo", im_out)

cv2.waitKey(0)