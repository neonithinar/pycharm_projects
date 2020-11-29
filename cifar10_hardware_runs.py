import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train= X_train_full[5000:]
X_valid= X_train_full[:5000]
y_train= y_train_full[5000:]
y_valid= y_train_full[:5000]

print(X_train.shape)

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

neurons = 80
layers = 15
model_name = "my_cifar10_selu_model_{}_layers_{}.h5".format(layers, neurons)


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape= [32, 32, 3]))
for _ in range(layers):
  model.add(keras.layers.Dense(neurons, kernel_initializer = 'lecun_normal', activation = 'selu'))

model.add(keras.layers.Dense(10, activation= 'softmax'))


optimizer = keras.optimizers.Nadam(lr= 7e-4)

model.compile(loss= 'sparse_categorical_crossentropy', optimizer= optimizer, metrics = ['accuracy'])
#path = os.path.join(os.curdir, "my_cifar10_selu_model.h5")
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(model_name, save_best_only=True)
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "my_cifar10_logs", "run_selu_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_valid_scaled = (X_valid - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

history = model.fit(X_train_scaled, y_train, epochs=100,
                    validation_data=(X_valid_scaled, y_valid), callbacks=callbacks)

pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model = keras.models.load_model(model_name)
print("After loading the model, and evaluating on the same validation dataset, Validation accuracy becomes")
model.evaluate(X_valid_scaled, y_valid)

# training force stopped at 28 epochs. low system memory. try reducing batch size