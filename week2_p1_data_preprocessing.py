# author: https://github.com/Luvata
# reimplementation and comments: jeiyoon
import setproctitle
import os

# pip install googledrivedownloader
from google_drive_downloader import GoogleDriveDownloader as gdd

import numpy as np
import random
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt

import imageio
from sklearn.utils import shuffle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setproctitle.setproctitle("[k4ke] meta_learning")

# Need to download the Omniglot dataset
if not os.path.isdir('./omniglot_resized'):
    gdd.download_file_from_google_drive(file_id = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                        dest_path = './omniglot_resized.zip',
                                        unzip = True)

assert os.path.isdir('./omniglot_resized')

# labels_images = get_images(select_classes, one_hot_labels, self.num_samples_per_class, shuffle=False)
def get_images(paths, labels, nb_samples = None, shuffle = True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels
    Args:
        paths: A list of character folders -> select_classes
        labels: list or numpy array of same length as paths -> one_hot_labels
        nb_samples: Number of images to retrieve per character -> self.num_samples_per_class
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    images_labels = [(i, os.path.join(path, image))
                    for i, path in zip(labels, paths)
                    for image in sampler(os.listdir(path))]

    if shuffle:
        # random.shuffle
        # https://wikidocs.net/79
        random.shuffle(images_labels)

    return images_labels

# test_images.append(image_file_to_array(img_path, 784))
def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename -> img_path
        dim_input: Flattened shape of image -> 784
    Returns:
        1 channel image
    """
    import imageio
    # misc.imread(filename)
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
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

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        # np.prod: return the product of array elements over a given axis
        # https://aigong.tistory.com/47
        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))
                             ]

        random.seed(1)
        # random.shuffle: list shuffle
        random.shuffle(character_folders)

        num_val = 100
        num_train = 1100

        self.metatrain_characer_folders = character_folders[: num_train]
        self.metaeval_character_folders = character_folders[
            num_train: num_train + num_val
        ]
        self.metatest_character_folders = character_folders[
            num_train + num_val:
        ]

    def sample_batch(self, batch_type, batch_size):
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

        all_image_batches = []
        all_label_batches = []

        for batch_idx in range(batch_size):
            # [1. Sample N different classes]
            # from either the specified train, test or validation folders

            # random.sample
            # https://docs.python.org/3/library/random.html
            select_classes = random.sample(folders, self.num_classes)

            # [2. Load K images per class and collect the associated labels]
            one_hot_labels = np.identity(self.num_classes)
            # DO NOT SHUFFLE THE ORDER OF EXAMPLES ACROSS CLASSES
            # IT MAKES THE OPTIMIZATION MUCH HARDER
            labels_images = get_images(select_classes, one_hot_labels, self.num_samples_per_class, shuffle = False)

            train_images, train_labels = [], []
            test_images, test_labels = [], []

            for sample_idx, (label, img_path) in enumerate(labels_images):
                # Take the first image of each class (index is 0, N, 2N, ...) to test_set
                if sample_idx % self.num_samples_per_class == 0:
                    test_images.append(image_file_to_array(img_path, 784))
                    test_labels.append(label)
                else:
                    train_images.append(image_file_to_array(img_path, 784))
                    train_labels.append(label)

            # Now we shuffle train & test, then concatenate them together
            train_images, train_labels = shuffle(train_images, train_labels)
            test_images, test_labels = shuffle(test_images, test_labels)

            # [3. Format the data and return two numpy matrices]
            # One of flatted images with shape [B, K, N, 784]
            # and one of one-hot labels [B, K, N, N]

            # np.vstack
            # https://rfriend.tistory.com/352
            # [K, N, N]
            labels = np.vstack(train_labels + test_labels).reshape((-1, self.num_classes, self.num_classes))
            # [K, N, 784]
            images = np.vstack(train_images + test_images).reshape((self.num_samples_per_class, self.num_classes, -1))

            all_image_batches.append(images)
            all_label_batches.append(labels)

        # 3. Return two numpy array [B, K, N, 784] and one-hot labels [B, K, N, N]
        # np.stack
        # https://everyday-image-processing.tistory.com/87
        all_image_batches = np.stack(all_image_batches)
        all_label_batches = np.stack(all_label_batches)

        return all_image_batches.astype(np.float32), all_label_batches.astype(np.float32)

def main(num_classes = 5, num_samples = 1, meta_batch_size = 16, random_seed = 1234):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    data_generator = DataGenerator(num_classes, num_samples + 1)

    # just for debug. should delete this
    _, _ = data_generator.sample_batch('train', meta_batch_size)

if __name__ == "__main__":
    results = main(num_classes = 5, num_samples = 1, meta_batch_size = 16, random_seed = 1234)


