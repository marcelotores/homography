# python coord.py imagem.jpg
import cv2
import sys

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        pontos.append((str(x), str(y)))
        font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
    cv2.imshow('Imagem', img)

    # checking for right mouse clicks

    if event == cv2.EVENT_RBUTTONDOWN:

        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]

        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    im = sys.argv[1]
    pontos = []

    img = cv2.imread(im)
    # resized_image = cv2.resize(img, (800, 598))
    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    print(pontos)
    cv2.destroyAllWindows()
