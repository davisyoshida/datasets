"""Adapted with changes for Python 3 compatibility from: https://github.com/tensorflow/models/blob/master/im2txt/im2txt/data/build_mscoco_data.py"""
from collections import Counter, defaultdict
import json
from operator import itemgetter
import os
from pathlib import Path
import pickle
import random
import sys
from threading import Thread

import nltk
import numpy as np
import tensorflow as tf


DATASETS_DIR = os.getenv('TF_DATASET_DIR', 'datasets')
DATASET_NAME = 'mscoco'
TF_DIRNAME = 'tf'
VOCAB_FILE = 'vocab.pkl'

DATA_DIR = os.path.join(DATASETS_DIR, DATASET_NAME)


MIN_WORD_COUNT = 4
START_WORD = '<S>'
END_WORD = '</S>'


IMAGE_FEATURE = 'image/data'
CAPTION_FEATURE = 'image/caption_ids'


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(bytes(v, encoding='utf-8')) for v in values])


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _preprocess_caption(caption):
    result = [START_WORD]
    result.extend(nltk.word_tokenize(caption.lower()))
    result.append(END_WORD)
    return result

def _get_vocab(captions):
    word_counts = Counter()
    for c in captions:
        word_counts.update(c)
    print(f'{len(word_counts)} distinct words')

    new_counts = [(w, c) for w, c in word_counts.items() if c >= MIN_WORD_COUNT]
    new_counts.sort(key=itemgetter(1), reverse=True)
    print(f'Kept {len(new_counts)} words')

    vocab = {w_count[0]: i for (i, w_count) in enumerate(new_counts)}
    return vocab

def _to_example(image, decoder, vocab):
    im_id, filename, caption = image
    with open(filename, 'rb') as f:
        im_data = f.read()

    try:
        decoder.decode_jpeg(im_data)
    except AssertionError:
        print(f'Skipping image {filename} because of invalid data')
        return

    context = tf.train.Features(feature={
        'image/image_id': _int64_feature(im_id),
        IMAGE_FEATURE: _bytes_feature(im_data)})

    unk_id = len(vocab)
    caption_as_ids = [vocab.get(w, unk_id) for w in caption]

    feature_lists = tf.train.FeatureLists(feature_list={
        CAPTION_FEATURE: _bytes_feature_list(caption),
        'image/caption_ids': _int64_feature_list(caption_as_ids)})

    return tf.train.SequenceExample(context=context, feature_lists=feature_lists)

def _make_image_protos(thread_index, num_threads, num_shards, vocab, decoder, images, out_path):
    shards_per_thread = num_shards // num_threads
    shard_splits = np.linspace(0, len(images), shards_per_thread + 1).astype(int)

    im_count = 0
    for shard in range(shards_per_thread):
        shard_index = thread_index * (shards_per_thread) + shard
        shard_path = out_path / f'shard-{shard_index:05d}-of-{num_shards:05d}'
        writer = tf.python_io.TFRecordWriter(str(shard_path.resolve()))

        shard_count = 0
        for im in images[shard_splits[shard]:shard_splits[shard + 1]]:
            example = _to_example(im, decoder, vocab)
            if example is not None:
                writer.write(example.SerializeToString())
                im_count += 1
                shard_count += 1

                if im_count % 1000 == 0:
                    print(f'Thread {thread_index}: Processed {im_count} of {len(images)} images')
                    sys.stdout.flush()

        writer.close()
        print(f'Thread {thread_index}: Wrote {shard_count} images to {shard_path}')
        sys.stdout.flush()

    print(f'Thread {thread_index}: Wrote {im_count} images to {shards_per_thread} shards')
    sys.stdout.flush()


def make_records(split='train', num_shards=150, num_threads=10):
    out_path = Path(DATA_DIR) / TF_DIRNAME / split
    if not out_path.exists():
        out_path.mkdir(parents=True)

    print('Loading captions')
    with open(os.path.join(DATA_DIR, 'captions', f'{split}.json')) as f:
        data = json.load(f)

    id_to_filename = {im['id']: im['file_name'] for im in data['images']}

    print('Preprocessing captions')
    id_to_captions = defaultdict(list)
    for im in data['annotations']:
        id_to_captions[im['image_id']].append(_preprocess_caption(im['caption']))

    print('Computing vocab')
    vocab = _get_vocab([c for caps in id_to_captions.values() for c in caps])
    with (Path(DATA_DIR) / TF_DIRNAME / VOCAB_FILE).open('wb') as f:
        pickle.dump(vocab, f)

    all_images = [(im_id, os.path.join(DATA_DIR, 'images', split, filename), caption)
        for im_id, filename in id_to_filename.items()
        for caption in id_to_captions[im_id]]

    random.shuffle(all_images)

    decoder = ImageDecoder()

    coord = tf.train.Coordinator()

    num_threads = min(num_threads, num_shards)
    assert num_shards % num_threads == 0
    slices = np.linspace(0, len(all_images), num_threads + 1).astype(int)

    print(f'Processing {len(all_images)} examples')
    threads = []
    for t_index in range(num_threads):
        args = (t_index,
                num_threads,
                num_shards,
                vocab,
                decoder,
                all_images[slices[t_index]:slices[t_index + 1]],
                out_path)
        thread = Thread(target=_make_image_protos, args=args)
        thread.start()
        threads.append(thread)

    coord.join(threads)

def _prefetch_input_data(
        reader,
        filenames,
        is_train,
        batch_size,
        values_per_shard,
        input_queue_capacity_factor=16,
        num_reader_threads=1,
        shard_queue_name='filename_queue',
        value_queue_name='input_queue'):

    if is_train:
        filename_queue = tf.train.string_input_producer(filenames, shuffle=True, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name=f'random_{value_queue_name}')
    else:
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[tf.string],
            name=f'fifo_{value_queue_name}')


    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))

    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    tf.summary.scalar(f'queue/{values_queue.name}/fraction_of_{capacity}_full', tf.cast(values_queue.size(), tf.float32) / capacity)

    return values_queue


def _read_example(example):
    context, sequence = tf.parse_single_sequence_example(
        example,
        context_features={
            IMAGE_FEATURE: tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            CAPTION_FEATURE: tf.FixedLenSequenceFeature([], dtype=tf.int64)
        })
    return context[IMAGE_FEATURE], sequence[CAPTION_FEATURE]


def _distort_image(image, thread_id):
    with tf.name_scope('flip_horizontal', values=[image]):
        image = tf.image.random_flip_left_right(image)
        color_ordering = thread_id % 2
        with tf.name_scope('distort_color', values=[image]):
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32 / 255)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.032)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_brightness(image, max_delta=32 / 255)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.032)

        image = tf.clip_by_value(image, 0.0, 1.0)

    return image

def _process_image(encoded_image, is_train, height, width, thread_id, resize_height=346, resize_width=346):
    def image_summary(name, image):
        if thread_id == 0:
            tf.summary.image(name, tf.expand_dims(image, 0))

    with tf.name_scope('decode', values=[encoded_image]):
        image = tf.image.decode_jpeg(encoded_image, channels=3)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_summary('original_image', image)

    image = tf.image.resize_images(image, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR)

    if is_train:
        image = tf.random_crop(image, [height, width, 3])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image_summary('resized_image', image)

    if is_train:
        image = _distort_image(image, thread_id)
    image_summary('final_image', image)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def _dynamic_pad_batch(images_and_captions, batch_size, queue_capacity):
    enqueue_list = []
    for image, caption in images_and_captions:
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(caption_length, 0)

        input_seq = tf.slice(caption, [0], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)
        enqueue_list.append([image, input_seq, indicator])

    images, input_seqs, mask = tf.train.batch_join(enqueue_list, batch_size=batch_size, capacity=queue_capacity, dynamic_pad=True, name='batch_and_pad')

    lengths = tf.add(tf.reduce_sum(mask, 1), 1)
    tf.summary.scalar('caption_length/batch_min', tf.reduce_min(lengths))
    tf.summary.scalar('caption_length/batch_max', tf.reduce_max(lengths))
    tf.summary.scalar('caption_length/batch_mean', tf.reduce_mean(lengths))

    return images, input_seqs, mask

def mscoco_input(batch_size=1, num_preprocess_threads=6, fold='train', is_train=False):
    file_dir = Path(DATA_DIR) / TF_DIRNAME / fold
    filenames = sorted(str(p) for p in file_dir.iterdir())

    with tf.name_scope('input'):
        reader = tf.TFRecordReader()
        input_queue = _prefetch_input_data(
            reader,
            filenames,
            is_train,
            batch_size,
            2500,
            input_queue_capacity_factor=16,
            num_reader_threads=1)

        images_and_captions = []
        for thread_id in range(num_preprocess_threads):
            serialized_example = input_queue.dequeue()
            encoded_image, caption = _read_example(serialized_example)
            image = _process_image(encoded_image, is_train, width=299, height=299, thread_id=thread_id)
            images_and_captions.append([image, caption])

        queue_capacity = (2 * num_preprocess_threads * batch_size)
        images, input_seqs, mask = _dynamic_pad_batch(images_and_captions, batch_size, queue_capacity)

    return images, input_seqs, mask



if __name__ == '__main__':
    make_records()
