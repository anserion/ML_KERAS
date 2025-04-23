# --------------------------------------------------
# создание и обучение искусственной нейронной сети
# типа многослойный персептрон в библиотеке Keras
# --------------------------------------------------
import numpy as np
import keras
from keras import layers
from keras import utils

# Создание модели
model = keras.Sequential()
# описываем входы нейросети (4 входа)
model.add(layers.Input(shape=(4,)))
# Полносвязные слои (3 слоя)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Резюме модели
model.summary()

# генерируем случайные входы-выходы для обучения
nSamples=10
X_train=np.random.rand(4*nSamples)
X_train=np.reshape(X_train,(nSamples,4,))
Y_train=np.random.randint(0,10,nSamples)
Y_train=utils.to_categorical(Y_train)

# Обучение модели
model.fit(X_train, Y_train, epochs=150, verbose=0)
# Оценка модели
print('model score: ',model.evaluate(X_train, Y_train))

# Включение модели в рабочий режим
predictions=model.predict(X_train)
for i,prediction in enumerate(predictions):
    print(f'predict: {np.argmax(prediction)}, real:{Y_train[i]}')
