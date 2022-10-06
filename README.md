
# AlexNet

AlexNet was the Convolutional Neural Network to win the ImageNet competition and is till date the most cited ML research papers , with around 80000 citations in october of 2022.

In This Repo we attempt at constructing an AlexNet architecture from scratch using tensorflow and Python , the implementation will be Non-Object Oriented.

Paper Link :https://noisrucer.github.io/paper/AlexNet/
# Architecture Outline
![AlexNet-architecture-Includes-5-convolutional-layers-and-3-fully-connected-layers](https://user-images.githubusercontent.com/111745916/194263535-c8752a39-2b05-4ac5-953d-142d99a67d04.png)

# Code

Importing Neccessary Libraries
```
import tensorflow as tf
from keras import layers
from keras import models
from keras import losses
from keras import datasets
```

Loading up the Data for training 

```
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_train = tf.repeat(x_train, 3, axis=3)
x_test = tf.repeat(x_test, 3, axis=3)
```

Splitting the Data into training and Validation sets

```
x_val = x_train[-2000:,:,:,:]
y_val = y_train[-2000:]
x_train = x_train[:-2000,:,:,:]
y_train = y_train[:-2000]
```

Model Construction and Summarization 
```
model = models.Sequential()
model.add(layers.Resizing(224, 224, interpolation="bilinear", 
input_shape=x_train.shape[1:]))
model.add(layers.Conv2D(96, 11,strides=4,padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(256, 5,strides=4,padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(384,3,strides=4,padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
-------------------------------------------------------------
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resizing_1 (Resizing)       (None, 224, 224, 3)       0         
                                                                 
 conv2d_5 (Conv2D)           (None, 56, 56, 96)        34944     
                                                                 
 lambda_2 (Lambda)           (None, 56, 56, 96)        0         
                                                                 
 activation_5 (Activation)   (None, 56, 56, 96)        0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 27, 27, 96)       0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 7, 7, 256)         614656    
                                                                 
 lambda_3 (Lambda)           (None, 7, 7, 256)         0         
                                                                 
 activation_6 (Activation)   (None, 7, 7, 256)         0         
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 3, 3, 256)        0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           (None, 1, 1, 384)         885120    
                                                                 
 activation_7 (Activation)   (None, 1, 1, 384)         0         
                                                                 
 conv2d_8 (Conv2D)           (None, 1, 1, 384)         1327488   
                                                                 
 activation_8 (Activation)   (None, 1, 1, 384)         0         
                                                                 
 conv2d_9 (Conv2D)           (None, 1, 1, 256)         884992    
                                                                 
 activation_9 (Activation)   (None, 1, 1, 256)         0         
                                                                 
 flatten_1 (Flatten)         (None, 256)               0         
                                                                 
 dense_3 (Dense)             (None, 4096)              1052672   
                                                                 
 dropout_2 (Dropout)         (None, 4096)              0         
                                                                 
 dense_4 (Dense)             (None, 4096)              16781312  
                                                                 
 dropout_3 (Dropout)         (None, 4096)              0         
                                                                 
 dense_5 (Dense)             (None, 10)                40970     
                                                                 
=================================================================
Total params: 21,622,154
Trainable params: 21,622,154
Non-trainable params: 0
```
Compiling and Training Phase of the Model

```
model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))
--------------------------------------------------------------------------------------------------------------------------------------
Epoch 1/5
907/907 [==============================] - 107s 117ms/step - loss: 0.4213 - accuracy: 0.8502 - val_loss: 0.0669 - val_accuracy: 0.9830
Epoch 2/5
907/907 [==============================] - 106s 117ms/step - loss: 0.0976 - accuracy: 0.9751 - val_loss: 0.1103 - val_accuracy: 0.9705
Epoch 3/5
907/907 [==============================] - 106s 117ms/step - loss: 0.0754 - accuracy: 0.9806 - val_loss: 0.0799 - val_accuracy: 0.9775
Epoch 4/5
907/907 [==============================] - 106s 117ms/step - loss: 0.0614 - accuracy: 0.9849 - val_loss: 0.0441 - val_accuracy: 0.9920
Epoch 5/5
907/907 [==============================] - 106s 117ms/step - loss: 0.0477 - accuracy: 0.9880 - val_loss: 0.0425 - val_accuracy: 0.9870
```
Evaluation of the model on The test Set.
```
model.evaluate(x_test,y_test)
----------------------------------------------------------------------------
313/313 [==============================] - 7s 22ms/step - loss: 0.0810 - accuracy: 0.9796
[0.08097803592681885, 0.9796000123023987]
```

