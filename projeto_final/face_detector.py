import numpy as np
from   cv2 import cv2

def crop_face(face, coordinates):
    x = coordinates[0]
    y = coordinates[1]
    w = coordinates[2]
    h = coordinates[3]

    return face[y:y+h, x:x+w]

class face_detector:
    def __init__(self, face_training_path='arquivos_treinamento/haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(face_training_path)

    # scaleFactor  - Especifica quanto a img eh reduzida para cada escala gerada (verificar funcionamento do algoritmo) 
    # minNeighbors - Num de vizinhos que cada retangulo deve ter para a analise. Valores grandes resultam em menos deteccoes, mas com mais qualidade
    # position     - front, left, right, up, down, burned. Deve-se ajustar o arquivo de treinamento de acordo com a posicao passada
    def detect(self, rgb_pictures_as_array, position, scaleFactor=2, minNeighbors=1):
        cropped_faces = {}

        for key, value in rgb_pictures_as_array.items():
            face  = value[position]
            gray  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            face_coordinates = self.face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors) 

            # TODO: Tratar quando mais de um rosto for detectado
            if face_coordinates is None or len(face_coordinates) != 1:
                continue

            coordinates = face_coordinates[0].tolist()

            cropped_faces[key] = crop_face(face, coordinates)

        return cropped_faces