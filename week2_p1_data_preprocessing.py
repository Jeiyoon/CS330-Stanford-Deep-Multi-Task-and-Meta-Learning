# author: https://github.com/LecJackS
# reimplementation and comments: jeiyoon
import setproctitle
import os

# pip install googledrivedownloader
from google_drive_downloader import GoogleDriveDownloader as gdd

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import imageio
from sklearn.utils import shuffle


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setproctitle.setproctitle("[k4ke] meta_learning_test")

# Need to download the Omniglot dataset
if not os.path.isdir('./omniglot_resized'):
    gdd.download_file_from_google_drive(file_id = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                        dest_path = './omniglot_resized.zip',
                                        unzip = True)

assert os.path.isdir('./omniglot_resized')

decoder = lambda x: image if type(image) is not bytes else image.decode('UTF-8')

# get_images(n_classes, range(N), nb_samples = K, shuffle = False)
def get_images(paths, labels, nb_samples = None, shuffle = True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels
    Args:
        paths: A list of character folders -> select_classes -> N
        labels: list or numpy array of same length as paths -> one_hot_labels
        nb_samples: Number of images to retrieve per character -> self.num_samples_per_class -> K
    Returns:
        List of (label, image_path) tuples
    """
    # nb_samples = 2
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    # labels: [[1,0,0,0,0], ...]
    # paths: ['./omniglot_resized/Grantha/character19', ...]
    # len(images_labels) = 5 * 2(nb_samples) = 10
    images_labels = [(i, os.path.join(path, image.decode('UTF-8')))
                    for i, path in zip(labels, paths)
                    for image in sampler(os.listdir(path))]

    if shuffle:
        # random.shuffle
        # https://wikidocs.net/79
        random.shuffle(images_labels)

    return images_labels


# image_file_to_array(im, I)
def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename -> img_path
        dim_input: Flattened shape of image -> 784
    Returns:
        1 channel image
    """
    image = imageio.imread(filename) # (28, 28)
    image = image.reshape([dim_input]) # (784,)
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image

    return image

class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omiglot data
    A "class" is considered a class omniglot digits.
    """
    def __init__(self, num_classes, num_samples_per_class, config = {}):
        """
        Args:
            num_classes: Number of classes for classification
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        # data_folder = './omniglot_resized'
        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        # np.prod: return the product of array elements over a given axis
        # https://aigong.tistory.com/47
        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        # e.g.) character_folders = ['./omniglot_resized/Hebrew/character05', ...]
        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))
                             ]

        np.random.seed(1)
        # random.shuffle: list shuffle
        np.random.shuffle(character_folders)

        num_val = 100
        num_train = 1100

        self.metatrain_characer_folders = character_folders[: num_train]
        self.metaeval_character_folders = character_folders[
            num_train: num_train + num_val
        ]
        self.metatest_character_folders = character_folders[
            num_train + num_val:
        ]

    def sample_batch(self, batch_type, batch_size, shuffle = True):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] / 784 = 28 * 28
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_characer_folders
        elif batch_type == "val":
            folders = self.metaeval_character_folders
        else:
            folders = self.metatest_character_folders

        # images: [B, K, N, 784] = [B, K, N, I]
        # labels: [B, K, N, N]
        B = batch_size
        K = self.num_samples_per_class
        N = self.num_classes
        I = self.dim_input

        all_image_batches = []
        all_label_batches = []

        for batch in range(B):
            # [1. Sample N different classes]
            # from either the specified train, test or validation folders

            # e.g. n_classes = ['./omniglot_resized/Grantha/character19', ...]
            n_classes = np.random.choice(folders, N, replace=False)

            # [2. Load K images per class and collect the associated labels]
            # DO NOT SHUFFLE THE ORDER OF EXAMPLES ACROSS CLASSES
            # IT MAKES THE OPTIMIZATION MUCH HARDER

            # e.g.) [(0, './omniglot_resized/ULOG/character14/1611_17.png'), ...]
            tuples = get_images(n_classes, range(N), nb_samples = K, shuffle = False)

            # I = self.dim_input = 784 -> images: [B, K, N, 784] = [B, K, N, I]
            images = [image_file_to_array(im, I) for lb, im in tuples] # images: [(784,), (784), ... ] / len = 2 * 5
            images = np.stack(images) # shape: (10, 784) / ndarray
            images = np.reshape(images, (N, K, I)) # shape: (5, 2, 784) / ndarray / (N, K, I)

            # e.g.) labels = [0 0 1 1 2 2 3 3 4 4]
            labels = np.asarray([lb for lb, im in tuples]) # shape: (10, ) / ndarray
            labels = np.reshape(labels, (N, K)) # shape: (5, 2) / e.g.) [[0 0], [1 1], [2 2], [3 3], [4 4]]
            labels = np.eye(N)[labels] # shape: (5, 2, 5) / (N, K, N) / e.g.) [[[1. 0. 0. 0. 0.],  [1. 0. 0. 0. 0.]],, [[0. 1. 0. 0. 0.],  [0. 1. 0. 0. 0.]],, [[0. 0. 1. 0. 0.],  [0. 0. 1. 0. 0.]],, [[0. 0. 0. 1. 0.],  [0. 0. 0. 1. 0.]],, [[0. 0. 0. 0. 1.],  [0. 0. 0. 0. 1.]]]

            labels = np.swapaxes(labels, 0, 1) # [K, N, I]
            images = np.swapaxes(images, 0, 1) # [K, N, N]

            all_image_batches.append(images)
            all_label_batches.append(labels)

        all_image_batches = np.stack(all_image_batches)
        all_label_batches = np.stack(all_label_batches)

        return all_image_batches, all_label_batches
    

if __name__ == "__main__":
    num_classes = 5 # N
    num_samples_per_class = 3 # K

    data = DataGenerator(num_classes, num_samples_per_class)

    images, labels = data.sample_batch('train', batch_size = 2, shuffle = False)

    print("Batch of images of shape: ", images.shape) # B K N I
    print("Batch of labels of shape: ", labels.shape) # B K N N

    # for i in range(num_classes):
    #     print(labels[0, 0, i])

    # print("First meta-batch")
    print("Second meta-batch")

    plt.figure(figsize = (16, 16))
    count = 0

    for cl in range(num_classes):
        for sa in range(num_samples_per_class):
            plt.subplot(num_classes, num_samples_per_class, count + 1)
            plt.title("Class {} - Example {}".format(cl, sa))
            # image = images[0, sa, cl].reshape((28, 28))
            image = images[1, sa, cl].reshape((28, 28))
            plt.imshow(image)
            plt.axis('off')
            count += 1
    plt.show()

    print("# of samples per class: ", data.num_samples_per_class)
    print("# of classes: ", data.num_classes)
    print("image size: ", data.img_size)
    print("input dimension: ", data.dim_input)
    print("output dimension: ", data.dim_output)

    print("# of meta train folders: ", len(data.metatrain_characer_folders))
    print("# of meta val folders: ", len(data.metaeval_character_folders))
    print("# of meta test folders: ", len(data.metatest_character_folders))

    print(data.metatrain_characer_folders[0:5])
    print(data.metatest_character_folders[-5:])

    # show image paths
    classes = np.asarray(data.metatest_character_folders[0:3])
    labels = [0, 1, 2]
    print(get_images(classes, labels, 1))

    # extracting family name
    print([os.path.basename(os.path.split(family)[0]) for family in data.metatest_character_folders[0:3]])

    # plot a characer directly from file
    plt.imshow(image_file_to_array('./omniglot_resized/N_Ko/character12/0815_17.png', 784).reshape((28, 28)))
    plt.show()






