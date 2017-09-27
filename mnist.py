import gzip
import os
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
from urllib.request import urlopen

DATASET_NAME = 'mnist'

TRAIN_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
TEST_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

DATA_DIR = os.getenv('TF_DATASET_DIR', 'datasets')

WIDTH = 28
HEIGHT = 28

def path_from_url(url):
    return Path(DATA_DIR) / DATASET_NAME / Path(url.split('/')[-1].split('.')[0])

def maybe_download(url):
    f_path = path_from_url(url)
    if not f_path.exists():
        f_path.parent.mkdir(exist_ok=True, parents=True)
        with urlopen(url) as u, f_path.open(mode='wb') as f:
            buf = BytesIO(u.read())
            unzipped = gzip.GzipFile(fileobj=buf)
            f.write(unzipped.read())

def bytes_to_int(bs):
    return sum(b << (8 * i) for i, b in enumerate(bs[::-1]))

def read_examples_file(f_path):
    with f_path.open('rb') as f:
        f.read(4)
        n_images = bytes_to_int(f.read(4))
        f.read(8) # Throw away width and height
        for _ in range(n_images):
            yield list(f.read(WIDTH * HEIGHT))

def read_labels_file(f_path):
    with f_path.open('rb') as f:
        f.read(4)
        n_labels = bytes_to_int(f.read(4))
        for _ in range(n_labels):
            yield f.read(1)[0]

def make_records():
    for dirname, example_url, label_url in (('train', TRAIN_URL, TRAIN_LABEL_URL), ('test', TEST_URL, TEST_LABEL_URL)):
        maybe_download(example_url)
        maybe_download(label_url)
        data_path = Path(DATA_DIR) / DATASET_NAME / dirname
        data_path.mkdir(parents=True, exist_ok=True)

        writer = None
        shard = 0
        for count, (example, label) in enumerate(zip(
                read_examples_file(path_from_url(example_url)),
                read_labels_file(path_from_url(label_url)))):
            if count % 5000 == 0:
                if writer is not None:
                    writer.close()
                writer = tf.python_io.TFRecordWriter(str(data_path / 'shard{}'.format(shard)))
                shard += 1
            proto = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(int64_list=tf.train.Int64List(value=example))}))
            writer.write(proto.SerializeToString())
        writer.close()

def read_example(filename_queue):
    reader = tf.TFRecordReader()
    _, ex = reader.read(filename_queue)
    features = tf.parse_single_example(
        ex,
        features={
            'image': tf.FixedLenFeature(shape=WIDTH * HEIGHT, dtype=tf.int64),
            'label': tf.FixedLenFeature([], dtype=tf.int64)})
    image = tf.reshape(features['image'], shape=(WIDTH, HEIGHT, 1))
    image = tf.cast(image, tf.float32) / 127.5 - 1

    label = features['label']
    return image, label


def mnist_input(batch_size=1, num_epochs=None, is_train=True, num_threads=4):
    file_dir = Path(DATA_DIR) / DATASET_NAME / ('train' if is_train else 'test')
    filenames = sorted(str(p) for p in file_dir.iterdir())
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, shuffle=is_train, num_epochs=num_epochs)
        example, label = read_example(filename_queue)
        examples, labels = (tf.train.shuffle_batch if is_train else tf.train.batch)(
            [example, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=max(5000, 10 * batch_size),
            **({'min_after_dequeue': max(2500, 5 * batch_size)} if is_train else {}))
        tf.summary.image('mnist_images', examples)

    return examples, labels

if __name__ == '__main__':
    make_records()
