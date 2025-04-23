# --------------------------------------------------
# создание и обучение одиночного нейрона (регресссия)
# в библиотеке Keras
# --------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers

# Создание модели
model = keras.Sequential()

# описываем входы нейросети (1-мерный сигнал, 1 вход)
model.add(layers.Input(shape=(1,)))

# одиночный нейрон, который будет тренироваться на
# генерацию значений функции y=kx+b
model.add(layers.Dense(units=1, activation='relu'))

# Компиляция модели
model.compile(optimizer='sgd', loss='mean_squared_error')

# Резюме модели
model.summary()

# Генерация входных данных для обучения
X_train = np.linspace(0, 1, 101)
# Генерация выходных данных для обучения
Y_train=2*X_train+1

# Обучение модели
model.fit(X_train, Y_train, epochs=200, verbose=0)
# Оценка модели
print('model train error: ',model.evaluate(X_train, Y_train))

# Генерация входных данных для теста
X_test = np.random.rand(10)

# Прямой проход нейросети на множестве X_test
Y_test = model.predict(X_test)

# Построение графика
plt.scatter(X_train, Y_train, color='blue', label='Train')
plt.scatter(X_test,Y_test, color='red', label='Predicted')
plt.legend()
plt.show()