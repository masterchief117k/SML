#T.Vinita
#2241022012
import numpy as np
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

np.random.seed(42)

# Load CIFAR-10 dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Define model
classier = Sequential()
classier.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
classier.add(LeakyReLU(alpha=0.3))
classier.add(Conv2D(64, padding='same', kernel_size=(3, 3)))
classier.add(LeakyReLU(alpha=0.3))
classier.add(MaxPooling2D(pool_size=(2, 2)))
classier.add(Dropout(0.25))
classier.add(Conv2D(128, kernel_size=(3, 3)))
classier.add(MaxPooling2D(pool_size=(2, 2)))
classier.add(Conv2D(128, kernel_size=(3, 3)))
classier.add(LeakyReLU(alpha=0.3))
classier.add(MaxPooling2D(pool_size=(2, 2)))
classier.add(Dropout(0.25))
classier.add(Flatten())  # Flatten the feature map tensor to 1D array
classier.add(Dense(1024))
classier.add(LeakyReLU(alpha=0.3))
classier.add(Dropout(0.5))
classier.add(Dense(10, activation='softmax'))
classier.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train model
history = classier.fit(X_train / 255.0, to_categorical(Y_train), batch_size=128,
                       shuffle=True,
                       epochs=250,
                       validation_data=(X_test / 255.0, to_categorical(Y_test)),
                       callbacks=[EarlyStopping(min_delta=0.01, patience=4)])

# Evaluate model
scores = classier.evaluate(X_test / 255.0, to_categorical(Y_test))
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Make predictions
Y_preds = classier.predict(X_test / 255.0)

# Print sample prediction
sample_id = 108
plt.figure(figsize=(4, 4))
plt.imshow(X_test[sample_id])
plt.axis('off')
print(Y_test[sample_id])
softmax_output = classier.predict(X_test[108].reshape((1,) + X_test[108].shape) / 255.)
print(softmax_output)
print(np.argmax(softmax_output))

# Print confusion matrix
confusion_matrix = confusion_matrix(Y_test, Y_preds.argmax(axis=1))
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.colorbar()
plt.show()