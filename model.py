import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras import activations, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, (60000, 28, 28, 1))
X_test = np.reshape(X_test, (10000, 28, 28, 1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#define model
model = Sequential()

model.add(Conv2D(32, (5,5), activation = activations.relu, padding = 'same', input_shape = (28, 28, 1)))
model.add(Conv2D(32, (5,5), activation = activations.relu, padding = 'same', input_shape = (28, 28, 1)))
model.add(MaxPooling2D((2,2), strides = 2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation = activations.relu, padding = 'same', input_shape = (28, 28, 1)))
model.add(Conv2D(64, (3,3), activation = activations.relu, padding = 'same', input_shape = (28, 28, 1)))
model.add(MaxPooling2D((2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = activations.relu))
model.add(Dropout(0.25))
model.add(Dense(10, activation = activations.softmax))

model.summary()
# fit model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metric = 'accuracy')
epoch = 30
batch_size = 112

model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = epoch, validation_data = (X_test, y_test))

model.save('./models/modelRD.h5')
model.save_weights('./models/weight_of_modelRD.h5')

# load_model at path('./models/modelRD.h5')
# load_weight at path('./models/weight_of_modelRD.h5')

# new_model = models.load_model('./models/modelRD.h5')
# new_model.load_weights('./models/weight_of_modelRD.h5')

