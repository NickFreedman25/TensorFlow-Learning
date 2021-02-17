import tensorflow
from tensorflow import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([7.0]))
