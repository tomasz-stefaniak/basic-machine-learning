import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# # UNCOMMENT THIS TO VIEW A SINGLE IMAGE
# IMAGE_INDEX = 0
# plt.figure()
# plt.imshow(train_images[IMAGE_INDEX])
# plt.colorbar()
# plt.show()


# # UNCOMMENT THIS TO VIEW IMAGES WITH LABELS
# NUMBER_OF_ROWS = 5
# NUMBER_OF_COLUMNS = 5
# NUMBER_OF_IMAGES = 25
# FIGURE_SIZE = (9, 9)
# plt.figure(figsize=FIGURE_SIZE)
# for index in range(NUMBER_OF_IMAGES):
#     plt.subplot(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, index + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[index], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[index]])
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0
