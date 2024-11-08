# scripts/predict.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar o modelo treinado
model = tf.keras.models.load_model('models/plant_recognition_model.h5')

# Função para prever a classe de uma imagem
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normaliza a imagem
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona a dimensão do batch

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return 'Mboa'
    else:
        return 'Tseke'

# Teste a predição com uma nova imagem
img_path = 'dataset/test/mboa/mboa1.jpeg'  # Caminho da imagem a ser testada
result = predict_image(img_path)
print(f'Predição: {result}')
