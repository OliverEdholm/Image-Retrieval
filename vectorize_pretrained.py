'''
Vectorize pretrained

Program for vectorizing images with the help of taking a vector from a layer in 
a pretrained Convolutional neural network.

Oliver Edholm, 14 years old 2017-03-22 12:16
'''
# imports
import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from six.moves import xrange
from tqdm import tqdm

from embedding.extraction import inception_v3
from embedding.extraction import inception_v4
from embedding.extraction import vgg
from embedding.preprocessing import inception_preprocessing
from embedding.preprocessing import vgg_preprocessing
from utils import configs
from utils.ops import load_image
from utils.vector_file_handler import VectorSaver
from utils.vector_file_handler import establish_vectors_folder

slim = tf.contrib.slim

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vectorizing images through' \
                                                 ' a layer from pretrained ' \
                                                 ' model.')
    parser.add_argument('--model_path', help='Path to pretrained model.')
    parser.add_argument('--vectors_path',
                        help='Path to folder where vectors will be saved.',
                        nargs='?', const='vectors', default='vectors')
    parser.add_argument('--model_type', nargs='?', const='InceptionV4',
                        default='InceptionV4',
                        help='Which type of architecture pretrained model ' \
                             'has. The architectures currently supported ' \
                             'are: InceptionV3, InceptionV4, VGG16 and VGG19.')
    parser.add_argument('--layer_to_extract', nargs='?', const='Mixed_7a',
                        default='Mixed_7a',
                        help='Which layer to extract from the model.')
    parser.add_argument('--images_path', nargs='?', const='images',
                        default='images', help='Path to images to vectorize.')
    ARGS = parser.parse_args()


# functions
def get_size(args):
    if args.model_type in ['InceptionV3', 'InceptionV4']:
        return configs.INCEPTION_IMAGE_SIZE
    elif args.model_type in ['VGG16', 'VGG19']:
        return configs.VGG_IMAGE_SIZE
    else:
        raise Exception('Unknown model type: {}'.format(args.model_type))


def get_vgg16_embedding(image_tensor, model_endpoint,
                        batch_size=configs.BATCH_SIZE):
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        _, endpoints = vgg.vgg_16(image_tensor, is_training=False,
                                  spatial_squeeze=False)
        model_output = endpoints[model_endpoint]

    return tf.stack([model_output[i]
                     for i in xrange(batch_size)])


def get_vgg19_embedding(image_tensor, model_endpoint,
                        batch_size=configs.BATCH_SIZE):
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        _, endpoints = vgg.vgg_19(image_tensor, is_training=False,
                                  spatial_squeeze=False)
        model_output = endpoints[model_endpoint]

    return tf.stack([model_output[i]
                     for i in xrange(batch_size)])


def get_inception_v3_embedding(image_tensor, model_endpoint,
                               batch_size=configs.BATCH_SIZE):
    with tf.contrib.slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        model_output, _ = inception_v3.inception_v3_base(image_tensor,
                                                         final_endpoint=model_endpoint)

    return tf.stack([tf.reshape(model_output[i], [-1])
                     for i in xrange(batch_size)])


def get_inception_v4_embedding(image_tensor, model_endpoint,
                               batch_size=configs.BATCH_SIZE):
    with tf.contrib.slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        model_output, _ = inception_v4.inception_v4_base(image_tensor,
                                                         final_endpoint=model_endpoint)

    return tf.stack([tf.reshape(model_output[i], [-1])
                     for i in xrange(batch_size)])


def get_batches(inputs, batch_size=configs.BATCH_SIZE):
    cur_batch = []
    idx = 0
    for item in inputs:
        cur_batch.append(item)

        if (idx + 1) % batch_size == 0:
            yield cur_batch
            cur_batch = []

        idx += 1

    if cur_batch:
        for _ in xrange(batch_size - len(cur_batch)):
            cur_batch.append(np.zeros(cur_batch[0].shape))

        yield cur_batch


def build_graph(args):
    logging.info('building graph')
    size = get_size(args)

    inps = [tf.placeholder(tf.float32, shape=[size, size, 3],
                           name='inp{}'.format(i + 1))
            for i in xrange(configs.BATCH_SIZE)]

    if args.model_type in ['InceptionV3', 'InceptionV4']:
        preprocessing_function = inception_preprocessing.preprocess_for_eval
    elif args.model_type in ['VGG16', 'VGG19']:
        preprocessing_function = vgg_preprocessing.preprocess_image

    preprocessed_images = tf.stack([preprocessing_function(image, size, size)
                                    for image in inps])

    if args.model_type == 'VGG16':
        embed_function = get_vgg16_embedding
    elif args.model_type == 'VGG19':
        embed_function = get_vgg19_embedding
    elif args.model_type == 'InceptionV3':
        embed_function = get_inception_v3_embedding
    else:  # InceptionV4
        embed_function = get_inception_v4_embedding

    return embed_function(preprocessed_images, args.layer_to_extract), inps


def main():
    image_paths = [os.path.join(ARGS.images_path, file_name)
                   for file_name in os.listdir(ARGS.images_path)]

    # using generators to save memory
    size = get_size(ARGS)
    images = (load_image(image_path, size=[size, size],
                         failure_image=np.zeros([size, size, 3]))
              for image_path in image_paths)
    image_batches = get_batches(images)

    vectorize_op, inps_placeholder = build_graph(ARGS)

    init = tf.global_variables_initializer()
    init_fn = slim.assign_from_checkpoint_fn(ARGS.model_path,
                                             slim.get_model_variables())
    logging.info('starting session')
    with tf.Session() as sess:
        sess.run(init)
        init_fn(sess)

        vectors_path = establish_vectors_folder(ARGS.vectors_path, ARGS, True)
        vector_saver = VectorSaver(vectors_path)

        length = np.floor(len(image_paths) / configs.BATCH_SIZE) + \
                 int(bool(len(image_paths) / configs.BATCH_SIZE))
        idx = 0
        logging.info('vectorizing')
        for image_batch in tqdm(image_batches, total=length):
            vectors = sess.run(vectorize_op,
                               feed_dict=dict(zip(inps_placeholder, image_batch)))

            for vector in vectors:
                vector_saver.add_vector(image_paths[idx], vector)

                idx += 1
                if idx == len(image_paths):
                    break

    logging.info('saved data at {}'.format(vectors_path))


if __name__ == '__main__':
    main()
