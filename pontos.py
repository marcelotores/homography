# Python program to explain cv2.circle() method

# importing cv2
import cv2

# path
path = 'homo.png'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

# Center coordinates
center_coordinates = (141, 131)
# pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])
# Radius of circle
radius = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
image = cv2.circle(image, center_coordinates, radius, color, thickness)

# Displaying the image
cv2.imshow(window_name, image)
cv2.waitKey(0)
