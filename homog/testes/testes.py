import cv2

img = '../../imagens/pas_eo/p4.jpg'

sift = cv2.SIFT_create()

# Retorna os n melhores pontos, mas não ordenados
sift2 = cv2.SIFT_create(4)

imagem = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

kps = sift.detect(imagem, None)
kps2 = sift2.detect(imagem, None)
kps3, des3 = sift.compute(imagem, kps)
kps4, des4 = sift.detectAndCompute(imagem, None)

#kps3, des3 = sift.detectAndCompute(imagem,None)
#Mask optional mask specifying where to look for keypoints. Not set by default.
#Keypoints If passed, then the method will use the provided vector of keypoints instead of detecting them, and the algorithm just computes their descriptors.


kps_4 = sorted(kps, key=lambda x: -x.response)[:4]

atributos = ['angle', 'class_id', 'convert', 'octave', 'overlap', 'pt', 'response', 'size']

#print(kps[0].pt)
#print(kps4[0].pt)
#print(type(des4[0]))
#print(kps2)
#print(kps_4)

#for i in range(len(kps2)):
#    print(kps2[i], ' - ', kps2[i].response)

#for i in range(len(kps_4)):
#    print(kps_4[i], ' - ', kps_4[i].response)

