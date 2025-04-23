# --------------------------------------------------
# создание и обучение искусственной нейронной сети
# сверточного типа в библиотеке Keras
# с загрузкой графических образов MNIST
# --------------------------------------------------
import numpy as np
import keras
from keras import Sequential
from keras import layers
from keras import utils

# Создаем модель сверточной нейронной сети
model = Sequential()
model.add(layers.Input(shape=(28,28,1,)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Загружаем тестовые данные MNIST
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Приводим тестовые данные к нужному формату и масштабируем
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
Y_train = utils.to_categorical(Y_train)

X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
Y_test = utils.to_categorical(Y_test)

# Обучение модели
model.fit(X_train, Y_train, epochs=3)

# Оценка модели
score = model.evaluate(X_test, Y_test)
print("Model score:", score[1])

# Включение модели в рабочий режим
predictions=model.predict(X_test)
for i,prediction in enumerate(predictions):
    print(f'predict: {np.argmax(prediction)}, real:{Y_test[i]}')
