# importing the module
import cv2
import sys

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resized_image, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
    cv2.imshow('image', resized_image)


    # checking for right mouse clicks

    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell

        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        print(str(b) + ',' + str(g) + ',' + str(r))
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

# driver function
if __name__ == "__main__":

    #h, w = 0

    im = sys.argv[1]
    # h = sys.argv[2]
    # w = sys.argv[3]

    #im = '/home/infra/Imagens/lab/45.jpeg'

    # reading the image
    img = cv2.imread(im)

    #print(w, ' ', h)
    #resized_image = cv2.resize(img, (598, 800))
    resized_image = img

    # displaying the image
    cv2.imshow('image', resized_image)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
