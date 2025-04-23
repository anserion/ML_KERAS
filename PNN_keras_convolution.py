# --------------------------------------------------
# создание и обучение искусственной нейронной сети
# сверточного типа в библиотеке Keras
# --------------------------------------------------
import numpy as np
import keras
from keras import layers
from keras import utils

nSamples=10

# Создание модели
model = keras.Sequential()

# описываем входы нейросети (изображение 32х32, 1 цветовой канал)
model.add(layers.Input(shape=[32,32,1,]))

# Первый свёрточный блок
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Второй свёрточный блок
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Третий свёрточный блок
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# переходим к одномерному типу входа
model.add(layers.Flatten())

# Полносвязные слои (3 слоя)
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(nSamples, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Резюме модели
model.summary()

# генерируем случайные изображения для обучения
X_train=np.random.rand(32*32*nSamples)
X_train=np.reshape(X_train,(nSamples,32,32,1))
Y_train=np.random.randint(0,10,nSamples)
Y_train=utils.to_categorical(Y_train)

# Обучение модели
model.fit(np.array(X_train), np.array(Y_train), epochs=100, verbose=0)
# Оценка модели
print('model score: ',model.evaluate(X_train, Y_train))

# Включение модели в рабочий режим
predictions=model.predict(X_train)
for i,prediction in enumerate(predictions):
    print(f'predict: {np.argmax(prediction)}, real:{Y_train[i]}')