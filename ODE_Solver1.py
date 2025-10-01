import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

"""
Creación del modelo para resolver una EDO a partir de un modelo de red ya 
creado. Se usará como herencia para este caso y no empezar a construir el 
modelo desde cero.
"""

class ODE(Sequential):
    # **kwargs significa "any extra keyword arguments"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name = "loss")
        self.mse = tf.keras.losses.MeanSquaredError()
        
    @property 
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data):
        batch = tf.shape(data)[0]
        min = tf.cast(tf.reduce_min, tf.float32) 
        # Convierte los datos del tensor que contiene el  mínimo de
        # un tensor en datos de tipo flotante 32.
        max = tf.cast(tf.reduce_max, tf.float32)
        # Convierte los datos del tensor que contiene el máximo de
        # un tensor en datos de tipo flotante 32.
        x = tf.random.uniform((batch, 1), min_val = min, max_val = max)
        
        with tf.GradientTape() as tape: 
            with tf.GradientTape() as tape2: 
                tape2.watch(x) # x es a la variable respecto a la cual se 
                # realizará la derivación automática.
                y = self(x, Training = True)
            dy = tape2.gradient(y,x)
            x_0 = tf.zeros(batch, 1)
            y_0 = self(x_0, training = True)
            
            eq = x*dy + y - (x**2)*tf.math.cos(x)
            initial_condition = 0
            
            loss = self.mse(0., eq) + self.mse(y_0, initial_condition)
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
model = ODE()

model.add(Dense(10, activation='tanh', input_shape=(1,)))
model.add(Dense(1, activation='tanh'))
model.add(Dense(1))


model.summary()  

model.compile(optimizer=Adam(learning_rate = 0.1),metrics=['loss'])

x=tf.linspace(-2,2,100)
history = model.fit(x,epochs=500,verbose=0)
plt.plot(history.history["loss"])