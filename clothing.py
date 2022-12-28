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

    model.fit(train_images, train_labels, epochs=2)

    _, test_accuracy = model.evaluate(
        test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_accuracy)

    probability_model = tf.keras.Sequential([model,
                                            tf.keras.layers.Softmax()])

    return probability_model


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

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(index, predictions_array, true_label):
    true_label = true_label[index]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# display_single_image()
# display_images_with_labels()
model = setup_model(train_images, train_labels, test_images, test_labels)

predictions = model.predict(test_images)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
