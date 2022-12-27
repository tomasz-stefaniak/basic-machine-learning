import tensorflow as tf

# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# X_train = X_train / 255.0
# X_test = X_test / 255.0


img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(img_width, img_height)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=3)
model.evaluate(X_test, y_test)
