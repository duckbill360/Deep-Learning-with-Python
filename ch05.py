# ch05
# Deep learning for computer vision
# First, build a basic convnet.
# A convnet takes as input tensors of shape (image_height, image_width, image_channels)
# (not including the batch dimension). Here for MNIST exmaples, (28, 28, 1).
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Adding a classifier on top of the convnet
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# you can observe the structure using model.summary()
print(model.summary())


# Let’s train our convnet on the MNIST digits.
from keras.datasets import mnist
from keras.utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_images = train_data.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_data.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
