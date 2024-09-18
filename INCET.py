from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, Normalization, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy

# from tensorflow.keras.models import Model
import tensorflow as tf
# from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam
from CellMalaryaDetection.ReadData import read_image_data
import matplotlib.pyplot as plt


path = "D:/0penThis/imageProcessing/expression/images/train"
r = read_image_data()
CONFIGURATION = {
  "LEARNING_RATE": 0.01,
  "N_EPOCHS": 1,
  "BATCH_SIZE": 128,
  "DROPOUT_RATE": 0.0,
  "IM_SIZE": 48,
  "REGULARIZATION_RATE": 0.0,
  "N_FILTERS": 6,
  "KERNEL_SIZE": 3,
  "N_STRIDES": 1,
  "POOL_SIZE": 2,
  "N_DENSE_1": 100,
  "N_DENSE_2": 10,
}
full_dataset = r.read(path,CONFIGURATION['IM_SIZE'], CONFIGURATION['IM_SIZE'], CONFIGURATION['BATCH_SIZE'], number_of_channels=1)

DATASET_SIZE = tf.data.experimental.cardinality(full_dataset).numpy()
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

full_dataset = full_dataset.shuffle(buffer_size = 10)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)


# plateau = 0
# def scheduler(epoch, lr):
#     learning_rate = lr
#     global plateau
#     if True:
#         plateau += 1
#     if plateau > 5:
#         learning_rate -= 0.75
#     return learning_rate
#
#
# scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
#
#
# plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_accuracy', factor=0.75, patience=5, verbose=1
# )



_input = Input((CONFIGURATION['IM_SIZE'],CONFIGURATION['IM_SIZE'],1))


x = Conv2D(filters=64, kernel_size=(3,3), activation="relu")(_input)
# local contrast normalization
x = MaxPooling2D((2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3,3), activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)


y = Conv2D(filters=64, kernel_size=(3,3), activation="relu")(_input)
# local contrast normalization
y = MaxPooling2D((2, 2))(y)

y = Conv2D(filters=64, kernel_size=(3,3), activation="relu")(y)
y = BatchNormalization()(y)
y = MaxPooling2D((2, 2))(y)

y = Flatten()(y)

concatinated = Concatenate(axis=1)([x, y])
dropout_ = Dropout(0.2, noise_shape=None)(concatinated)

# dense2 = Dense(4096, activation="relu")(x)
output = Dense(7, activation="softmax")(dropout_)

model = tf.keras.models.Model(inputs=_input, outputs=output)
model.summary()

# print(normalized_dataset.shape)
model.compile(optimizer = Adam(learning_rate = 0.01),
      loss = CategoricalCrossentropy(), metrics=['accuracy'],
      )

# history = model.fit(
#     train_dataset,
#     epochs = 5,#CONFIGURATION['N_EPOCHS'],
#     validation_data=val_dataset,
#     verbose = 1,
#     #callbacks=[plateau_callback]#scheduler_callback()]
#     )

history = model.fit(train_dataset, validation_data=val_dataset, epochs = 10, verbose = 1)
# model.predict(train_dataset.reshape(800, 9))
# model.predict(val_dataset.reshape(100, 9))
# history = model.fit(train_dataset, validation_data=val_dataset, epochs = 100, verbose = 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

test = model.evaluate(test_dataset)
print(test)