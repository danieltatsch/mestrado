import numpy             as np
import matplotlib.pyplot as plt
import random
import pickle
from cv2           import cv2
from rgb_loader    import rgb_loader
from face_detector import face_detector

rap3df_database_path = 'C:/Users/Avell/projects/rap3df-database/'
position             = 'front'
IMG_SIZE             = 48
rgb_loader           = rgb_loader(rap3df_database_path)
face_detector        = face_detector()

def generate_training_data(faces, indice):
    training_data = []

    for key, value in faces.items():
        class_num = list(faces).index(key)
        class_num = 0 if class_num == indice else 1

        for i in range(len(value)):
            img_grayscale = cv2.cvtColor(value[i], cv2.COLOR_RGB2GRAY)
            training_data.append([img_grayscale, class_num])

    random.shuffle(training_data)

    X = []
    y = []

    for features, label, in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return [X,y]

################### MAIN ######################
all_rgb_pictures = rgb_loader.get_all_rgb_pictures()
faces            = face_detector.detect2(all_rgb_pictures)

for i in range(len(faces.keys())):
    X, y = generate_training_data(faces, i)

    X_name = 'dados2/X-{}.pickle'.format(i)
    y_name = 'dados2/y-{}.pickle'.format(i)

    pickle_out = open(X_name, 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(y_name, 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()