import cv2
import numpy as np

if __name__ == '__main__' :

    # Read source image.
    im_src = cv2.imread('46.jpeg')
    # Four corners of the book in source image
    pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])

    # Read destination image.
    im_dst = cv2.imread('46.jpeg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    print(h)
    print(status)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    imSF = cv2.resize(im_src, (960, 540))
    imSD = cv2.resize(im_dst, (960, 540))
    imSS = cv2.resize(im_out, (960, 540))
    # Display images
    cv2.imshow("Source Image", imSF)
    cv2.imshow("Destination Image", imSD)
    cv2.imshow("Warped Source Image", imSS)

    cv2.waitKey(0)