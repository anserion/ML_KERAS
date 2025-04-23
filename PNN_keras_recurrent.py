# --------------------------------------------------
# создание и обучение искусственной нейронной сети
# рекуррентного типа (LSTM) в библиотеке Keras
# --------------------------------------------------
import numpy as np
import keras
from keras import layers

# Создание модели нейросети
model = keras.Sequential()
# описываем входы нейросети (4 входа)
model.add(layers.Input(shape=(10,1,)))
# рекуррентный слой
model.add(layers.LSTM(64))
# выходной нейрон
model.add(layers.Dense(1))
# Компиляция модели
model.compile(optimizer='adam', loss='mse')
# Резюме модели
model.summary()

# генерируем случайные входы-выходы для обучения
nSamples=5
X_train=np.random.rand(10*nSamples)
X_train=np.reshape(X_train,(nSamples,10,1,))
Y_train=np.random.randint(0,10,nSamples)

# Обучение модели
model.fit(X_train, Y_train, epochs=500, verbose=0)
# Оценка модели
print('model score: ',model.evaluate(X_train, Y_train))

# Включение модели в рабочий режим
predictions=model.predict(X_train)
for i,prediction in enumerate(predictions):
    print(f'predict: {prediction}, real:{Y_train[i]}')