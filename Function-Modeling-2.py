import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import matplotlib.pyplot as plt 
import math as mt

loss_tracker = keras.metrics.Mean(name="loss")

class Function(keras.Model):
    def train_step(self, data):
        batch_size = 10
        x = tf.random.uniform((batch_size,), minval=-1, maxval=1) 
        # Se crea un tensor que va desde -1 hasta 1 de forma uniforme
        eq = 1 + 2*tf.square(x) + 4*tf.pow(x, 3)


        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = keras.losses.mean_squared_error(y_pred,eq)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}

class PolyTransform(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(PolyTransform,self).__init__()
        
        self.num_outputs = num_outputs
        self.kernel = self.add_weight("kernel",
                                shape=[self.num_outputs]) 
        # Pesos que harán de coeficientes

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if (inputs.shape == ()):
            inputs=(inputs,)
        elif (len(inputs.shape)==1):
            inputs=tf.expand_dims(inputs, axis=1)
            
        modes = tf.concat([tf.ones_like(inputs), inputs, inputs**2, inputs**3],
                          axis=1)        
        return tf.tensordot(modes,self.kernel,1) 
    # Producto punto entre a_i * x^i 

inputs = keras.Input(shape=(1,))
x = PolyTransform(4)(inputs)

model = Function(inputs=inputs,outputs=x)
model.compile(optimizer=Adam(learning_rate=0.1), metrics=['loss'])

x = tf.linspace(-1,1,100)
history = model.fit(x,epochs=50,verbose=1)

x_testv = tf.linspace(-1,1,100)
a = model.predict(x_testv)

plt.plot(x_testv, a, label="Modelado")
plt.plot(x_testv, 1 + 2*x**2 + 4*x**3, label = "Función", color = "orange")
plt.legend()
plt.grid()
plt.show()





