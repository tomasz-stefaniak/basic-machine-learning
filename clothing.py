import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

PIXEL_VALUE_RANGE = 255.0


def display_single_image():
    image_index = 0
    plt.figure()
    plt.imshow(train_images[image_index])
    plt.colorbar()
    plt.show()


def display_images_with_labels():
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
        plt.xlabel(class_names[train_labels[index]])
    plt.show()


def setup_model(train_images, train_labels, test_images, test_labels):
    train_images = train_images / PIXEL_VALUE_RANGE
    test_images = test_images / PIXEL_VALUE_RANGE

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_accuracy = model.evaluate(
        test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_accuracy)

    probability_model = tf.keras.Sequential([model,
                                            tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    print(np.argmax(predictions[0]))


display_single_image()
display_images_with_labels()
setup_model(train_images, train_labels, test_images, test_labels)
