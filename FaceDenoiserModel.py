from FacePreprocess import data
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), padding='same', activation='elu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPool2D((3, 3), padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='elu'),
    tf.keras.layers.MaxPool2D((3, 3), padding='same'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='elu'),
    tf.keras.layers.MaxPool2D((3, 3), padding='same'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='elu'),
    tf.keras.layers.MaxPool2D((3, 3), padding='same'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2 * 2 * 128),
    tf.keras.layers.Reshape((2, 2, 128)),

    tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), padding='same', activation='elu', strides=(2, 2)),
    tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(8, 8), padding='same', activation='elu', strides=(4, 4)),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), padding='same', activation='elu', strides=(2, 2)),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(8, 8), padding='same', activation='elu', strides=(4, 4)),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(4, 4), padding='same', activation='elu', strides=(2, 2)),
    tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=(4, 4), padding='same', strides=(2, 2)),
])

split = int(len(data) * .8)
train = data[:split]
test = data[split:]

model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['MAE'])
model.fit(x= train, y= train, epochs=25, validation_data=(test, test), verbose=1)
model.save('saved_model/face_gen_3')
