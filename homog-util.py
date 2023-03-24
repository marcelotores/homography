import numpy as np


def convert_point(kp_image, good_matches):
    src_pts = np.float32([kp_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return src_pts
