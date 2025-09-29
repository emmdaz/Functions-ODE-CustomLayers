import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy

"""
This code works using the DataSet that can be found in: 
https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
"""


ds = r"/archive" 
train_data_dir = ds + "/train"  
test_data_dir = ds + "/test"

datagen = ImageDataGenerator(rescale=1./255)

train = datagen.flow_from_directory(
        train_data_dir,
        class_mode="categorical",
        subset="training",batch_size=3)


test = datagen.flow_from_directory(
        test_data_dir,
        class_mode="categorical",
        subset="training")

class Filtro(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Filtro, self).__init__(**kwargs)
        self.weights_rgb = tf.constant([0.299, 0.587, 0.114],
                                       dtype=tf.float32)

    def call(self, inputs):
        gray = tf.tensordot(inputs, self.weights_rgb, axes=[[3], [0]])
        gray = tf.expand_dims(gray, axis=-1)
        return gray

model = tf.keras.Sequential([
    Filtro(input_shape=(128, 128, 3))
])
model.summary()



