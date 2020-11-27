import os
import tensorflow    as     tf
from   rgb_loader    import rgb_loader
from   face_detector import face_detector
from   cv2           import cv2

rap3df_database_path               = 'C:/Users/Avell/projects/rap3df-database/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importa a img de teste, converte para escala de cinza, ajusta o tamanho e aplica o reshape
def prepare2(face):
    IMG_SIZE  = 48
    img_array = face/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

rgb_loader       = rgb_loader(rap3df_database_path)
all_rgb_pictures = rgb_loader.get_all_rgb_pictures()
face_detector    = face_detector()
faces            = face_detector.detect2(all_rgb_pictures)

test_pictures = []
for key, value in faces.items():
    img_gray      = rgb_loader.get_picture('burned', key)
    test_pictures.append(face_detector.face_detect(img_gray))

print("Carregando modelos...")

prediction = []
for i in range(len(test_pictures)):
    # cv2.imshow("nada", test_pictures[i])
    # cv2.waitKey()

    model_name = 'resultados_final/CNN-{}.model'.format(i)
    model      = tf.keras.models.load_model(model_name)
    result     = model.predict([prepare2(test_pictures[i])])
    prediction.append([result])
    
    if result[0][0] > 0.5:
        print(i)