import numpy as np
from   cv2 import cv2

class face_detector:
    def __init__(self, face_training_path='arquivos_treinamento/haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(face_training_path)

    # scaleFactor  - Especifica quanto a img eh reduzida para cada escala gerada (verificar funcionamento do algoritmo) 
    # minNeighbors - Num de vizinhos que cada retangulo deve ter para a analise. Valores grandes resultam em menos deteccoes, mas com mais qualidade
    # img          - Imagem em np.array
    def detect(self, img, scaleFactor=2, minNeighbors=1):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

        return (True, faces) if len(faces) != 0 else (False, None)

        # TODO: Gerar nova imagem de acordo com as coordernadas retornadas
