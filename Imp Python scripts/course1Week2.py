import tensorflow as tf
from tensorflow import keras

data = tf.keras.datasets.fashion_mnist

(train_data, train_labels) , (test_data, test_labels) = data.load_data()


import matplotlib.pyplot as plt
plt.imshow(train_data[1])

train_data = train_data/255

test_data = test_data/255

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024,activation = tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation = 'softmax')])

model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy')
model.fit(train_data,train_labels,epochs = 5)

model.evaluate(test_data, test_labels)

###############################################################################
#Implementing Callbacks

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True
            
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels),(test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation = tf.nn.),
                                    tf.keras.layers.Dense(10,activation = tf.nn.softmax)])
    
model.compile(optimizer = 'adam', loss = tf.losses.mean_squared_error, metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 10, callbacks = [callbacks])

