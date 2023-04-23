import cv2 as cv
from math import sqrt
import types

import numpy as np


def res_img(img, nova_largura):

    """ Recebe uma imagem, e a nova largura, e devolve a imagem redimensionada proporcionalmente """

    altura, largura, c = img.shape
    val = altura / largura
    nova_altura = nova_largura * val

    print(int(nova_altura), nova_largura)
    return cv.resize(img, (nova_largura, int(nova_altura)), interpolation=cv.INTER_AREA)

def distancia_entre_pontos(ponto_1, ponto_2):
    xA, yA = ponto_1
    xB, yB = ponto_2

    # Calculando a distância
    distAB = sqrt((xA - xB) ** 2) + ((yA - yB) ** 2)

    return distAB

def retangulo2(ponto_topo_esquerdo, ponto_base_direita, imagem):

    cv.rectangle(imagem, ponto_topo_esquerdo, ponto_base_direita, (0, 255, 0), 5)
    return imagem

def retangulo(ponto_topo_esquerdo, ponto_base_direita, imagem, invertePontos=False):

    if invertePontos:
        x_anterior = ponto_topo_esquerdo[0]
        ponto_topo_esquerdo[0] = ponto_base_direita[0]
        ponto_base_direita[0] = x_anterior

    cv.rectangle(imagem, ponto_topo_esquerdo, ponto_base_direita, (0, 255, 0), 5)
    return imagem

def circulo(ponto, raio, imagem):
    cv.circle(imagem, ponto, raio, (255, 0, 0), 3)
    return imagem

def reta2(ponto_1, ponto_2, imagem):

    cv.line(imagem, ponto_1, ponto_2, (0, 0, 255), 2)

    return imagem

def reta(ponto_1, ponto_2, imagem):

    #ponto_1 = [356, 224]
    #ponto_2 = [329, 270]

    x1, y1 = ponto_1
    x2, y2 = ponto_2

    # coeficiente angular
    m = (y2 - y1) / (x2 - x1)

    # escolhe um dos dois pontos e substitui para encontrar o coeficiente linear
    n = y1 - m * x1
    # após, foi encontrado a equação da reta

    maior_x = max(x1, x2)
    menor_x = min(x1, x2)

    while menor_x < maior_x:
        y = m * menor_x + n
        imagem = pontos(imagem, menor_x, int(y), 'R')
        menor_x+=1

    return imagem


def pontos(image, x, y, cor='B', tamanho=1):

    """ Recebe uma imagem, suas coordenas X, Y e coloca um ponto azul sobre a imagem """

    #image = cv.imread(imagem)

    center_coordinates = (x, y)
    radius = tamanho
    if cor == 'B':
        color = (255, 0, 0)
    elif cor == 'G':
        color = (0, 255, 0)
    elif cor == 'R':
        color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # Draw a circle with blue line borders of thickness of 2 px
    image = cv.circle(image, center_coordinates, radius, color, thickness)

    #cv.imshow('Imagem', image)
    #cv.imwrite('imagem_com_ponto.jpg', image)
    #cv.waitKey(0)

    return image


def linha(image, ponto_a, ponto_b, cor='B', tamanho=1):

    """ Recebe uma imagem, suas coordenas X, Y e coloca um ponto azul sobre a imagem """

    #image = cv.imread(imagem)

    center_coordinates = (x, y)
    radius = tamanho
    if cor == 'B':
        color = (255, 0, 0)
    elif cor == 'G':
        color = (0, 255, 0)
    elif cor == 'R':
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

def sift_correspondencias(imagem1, imagem2, qtd_match=0):
    """ Recebe duas imagens (numpy) e retorna uma tupla (kp1, kp2, correspondencias[]) """

    img1 = cv.imread(imagem1)
    img2 = cv.imread(imagem2)

    kp1, des1 = sift_detectores_e_descritores(img1)
    kp2, des2 = sift_detectores_e_descritores(img2)

    bf = cv.BFMatcher()
    correspondencias = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in correspondencias:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    if qtd_match != 0:
        image_out = cv.drawMatchesKnn(img1, kp1, img2, kp2, good[:qtd_match], None,
                                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        image_out = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
                                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #return kp1, kp2, good
    return image_out, good, kp1, kp2

def orb_correspondencias(imagem1, imagem2):

    """ Recebe duas imagens (numpy) e retorna uma tupla (kp1, kp2, correspondencias[]) """

    img1 = cv.imread(imagem1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(imagem2, cv.IMREAD_GRAYSCALE)

    # Detectores e descritores
    kp1, des1 = orb_detectores_e_descritores(img1)
    kp2, des2 = orb_detectores_e_descritores(img2)

    if type(des1) != types.NoneType and type(des2) != types.NoneType:

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        image_out = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return image_out, matches
    else:
        print('Não há características o suficientes para serem correspondidas!')
    #return kp1, kp2, matches
