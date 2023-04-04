import cv2 as cv

def pontos(image, x, y, cor):

    """ Recebe uma imagem, suas coordenas X, Y e coloca um ponto azul sobre a imagem """

    #image = cv.imread(imagem)
    center_coordinates = (x, y)
    radius = 1
    if cor == 'vermelho':
        color = (255, 0, 0)
    elif cor == 'verde':
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Draw a circle with blue line borders of thickness of 2 px
    image = cv.circle(image, center_coordinates, radius, color, thickness)

    #cv.imshow('Imagem', image)
    #cv.imwrite('imagem_com_ponto.jpg', image)
    #cv.waitKey(0)

    return image

def sift_detectores_e_descritores(imagem):

    """ Recebe uma imagem (numpy) e retorna uma tupla com os detectores e descritores """

    # Converte para cinza
    gray = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

    # Cria o objeto sift
    sift = cv.SIFT_create()

    # Pega os detectores e descritores
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des

def orb_detectores_e_descritores(imagem):

    """ Recebe uma imagem (numpy) e retorna uma tupla com os detectores e descritores """

    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(imagem, None)

    return kp, des

def sift_correspondencias(imagem1, imagem2):
    """ Recebe duas imagens (numpy) e retorna uma tupla (kp1, kp2, correspondencias[]) """

    kp1, des1 = sift_detectores_e_descritores(imagem1)
    kp2, des2 = sift_detectores_e_descritores(imagem2)

    bf = cv.BFMatcher()
    correspondencias = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in correspondencias:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return kp1, kp2, good


def orb_correspondencias(imagem1, imagem2):

    """ Recebe duas imagens (numpy) e retorna uma tupla (kp1, kp2, correspondencias[]) """

    # Detectores e descritores
    kp1, des1 = orb_detectores_e_descritores(imagem1)
    kp2, des2 = orb_detectores_e_descritores(imagem2)

    # Initiate ORB detector
    orb = cv.ORB_create()

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches
