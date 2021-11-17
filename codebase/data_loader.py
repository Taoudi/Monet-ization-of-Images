import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds


FILENAMES_MONET = tf.io.gfile.glob('../monet_tfrec/' + "*.tfrec")
FILENAMES_REAL = tf.io.gfile.glob('../photo_tfrec/' + "*.tfrec")

print("Train TFRecord Files:", len(FILENAMES_MONET))
print("Train TFRecord Files:", len(FILENAMES_REAL))

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1
IMAGE_SIZE = [256,256]

def decode_image(image,IMAGE_SIZE=256):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0

def read_tfrecord(example, labeled=False):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string)
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    image = normalize_img(image)

    return image


def load_dataset(filenames, labeled=False):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def data():
    return get_dataset(FILENAMES_REAL),get_dataset(FILENAMES_MONET)

#image_batch = next(iter(train_monet))


def show_batch(image_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        if n == 0:
            print(image_batch[n])

        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n]/2 +0.5)
        plt.axis("off")
    plt.show()


#show_batch(image_batch.numpy())

#print(train_dataset.tonumpy().shape)