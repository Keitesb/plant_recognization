# scripts/train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Caminho para o dataset
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Aumento de dados para treino
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normaliza as imagens para o intervalo [0, 1]
    rotation_range=30,          # Rotação aleatória
    width_shift_range=0.2,      # Translação horizontal
    height_shift_range=0.2,     # Translação vertical
    shear_range=0.2,            # Cortes aleatórios
    zoom_range=0.2,             # Zoom aleatório
    horizontal_flip=True,       # Flip horizontal
    fill_mode='nearest'         # Preenchimento após transformação
)

# Para a fase de teste, apenas normalização
test_datagen = ImageDataGenerator(rescale=1./255)

# Gerador de imagens
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Modelo simples de CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Para classificação binária
])

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Salvar o modelo treinado
model.save('models/plant_recognition_model.h5')
