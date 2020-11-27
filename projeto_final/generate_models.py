import numpy             as np
import matplotlib.pyplot as plt
import os
import random
import pickle
from cv2 import cv2
import time
from sklearn.utils import class_weight

import tensorflow as tf
from   tensorflow.keras.models    import Sequential
from   tensorflow.keras.layers    import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from   tensorflow.keras.callbacks import TensorBoard
from   sklearn.model_selection    import train_test_split
from   sklearn.metrics            import confusion_matrix, classification_report
from   sklearn.utils              import class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NAME = "face-recognition-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))

pickle_in = open('X.pickle', 'rb')
X         = pickle.load(pickle_in)
pickle_in = open('y.pickle', 'rb')
y         = pickle.load(pickle_in)

X = np.array(X)
y = np.array(y)
X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classes      = np.unique(y_train)
peso_classes = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
peso_classes = {0: peso_classes[0], 1: peso_classes[1]}

print("PESOS:")
print(peso_classes)

dense_layers = [0, 1, 2]
layer_sizes  = [32, 64, 128]
conv_layers  = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME        = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            
            model.add(Flatten())
            model.add(Dropout(0.3))

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
                model.add(Dropout(0.3))

            model.add(Dense(2))
            model.add(Activation('softmax'))

            model.compile(loss='sparse_categorical_crossentropy',
                        optimzer='adam',
                        metrics=['accuracy'])

            history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=30, callbacks=[tensorboard], class_weight=peso_classes, validation_data=(X_test, y_test))
            model.save('modelos_cnn3\{}'.format(NAME))