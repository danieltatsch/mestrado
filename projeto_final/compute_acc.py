import numpy             as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import json
import time

from cv2 import cv2
import tensorflow as tf
from   tensorflow.keras.models    import Sequential
from   tensorflow.keras.layers    import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from   tensorflow.keras.callbacks import TensorBoard
from   sklearn.model_selection    import train_test_split
from   sklearn.utils              import class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def classify(X, y, indice):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    peso_classes = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    print("PESOS: " + str(peso_classes))

    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    model.add(Dense(2))
    model.add(Activation("softmax"))

    model.compile(loss='sparse_categorical_crossentropy',
                optimzer='adam',
                metrics=['accuracy'])

    history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=30, class_weight=peso_classes, validation_data=(X_test, y_test))

    model.save('resultados_final\CNN-{}.model'.format(indice))

    acc      = [ float('%.5f' % elem) for elem in history.history['accuracy'] ]
    val_acc  = [ float('%.5f' % elem) for elem in history.history['val_accuracy'] ]
    loss     = [ float('%.5f' % elem) for elem in history.history['loss'] ]
    val_loss = [ float('%.5f' % elem) for elem in history.history['val_loss'] ]
    
    y_pred = model.predict(X_test)
    y_pred_final = []

    for i in y_pred:
        if i[1] > 0.5:
            y_pred_final.append(1)
        else:
            y_pred_final.append(0)

    return [acc, val_acc, loss, val_loss, y_test.tolist(), y_pred_final]

#### MAIN ####
precisao = {}

for i in range(72):
    print("ITERATION: "  + str(i))
    print('=================')

    X_name = 'dados2/X-{}.pickle'.format(i)
    y_name = 'dados2/y-{}.pickle'.format(i)

    pickle_in = open(X_name, 'rb')
    X         = pickle.load(pickle_in)
    pickle_in = open(y_name, 'rb')
    y         = pickle.load(pickle_in)

    X = np.array(X)
    X = X/255.0
    y = np.array(y)

    precisao[i]                 = {}
    precisao[i]['accuracy']     = []
    precisao[i]['val_accuracy'] = []
    precisao[i]['loss']         = []
    precisao[i]['val_loss']     = []
    precisao[i]['y_test']       = []
    precisao[i]['y_pred']       = []

    precisao[i]['accuracy'], precisao[i]['val_accuracy'], precisao[i]['loss'], precisao[i]['val_loss'], precisao[i]['y_test'], precisao[i]['y_pred'] = classify(X, y, i)

with open('ouput_final.json', 'w') as fp:
    json.dump(precisao, fp, indent=2)