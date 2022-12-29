import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def display_images_with_labels(train_images, train_labels):
    number_of_rows = 5
    number_of_columns = 5
    number_of_images = 25
    figure_size = (9, 9)
    plt.figure(figsize=figure_size)
    for index in range(number_of_images):
        plt.subplot(number_of_rows, number_of_columns, index + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(train_images[index], cmap=plt.cm.binary)
        plt.xlabel(train_labels[index])
    plt.show()


def train_model(x_train, y_train, x_test, y_test):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

    return model


def plot_image(index, predictions_array, true_label, img):
    true_label, img = true_label[index], img[index]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100*np.max(predictions_array),
                                         true_label),
               color=color)


def plot_value_array(index, predictions_array, true_label):
    true_label = true_label[index]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    current_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    current_plot[predicted_label].set_color('red')
    current_plot[true_label].set_color('blue')


def display_predictions(test_images, test_labels, predictions):
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for index in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*index+1)
        plot_image(index, predictions[index], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*index+2)
        plot_value_array(index, predictions[index], test_labels)
    plt.tight_layout()
    plt.show()


# display_images_with_labels(train_images=x_train, train_labels=y_train)

model = train_model(x_train, y_train, x_test, y_test)
predictions = model.predict(x_test)

display_predictions(x_test, y_test, predictions)
