'''
Ops

Module for general operations.

Oliver Edholm, 14 years old 2017-03-26 17:07
'''
# imports
import logging
from six.moves import cPickle

import numpy as np
from PIL import Image


# functions
def get_pkl_file(file_path):
    with open(file_path, 'rb') as pkl_file:
        return cPickle.load(pkl_file)


def save_pkl_file(data, file_path):
    with open(file_path, 'wb') as pkl_file:
        cPickle.dump(data, pkl_file)


def load_image(image_path, size=None, failure_image=None,
               allow_non_rgb=False):
    try:
        image = Image.open(image_path)
    except Exception as e:
        logging.warning('error loading image at "{}" with ' \
                        'exception "{}"'.format(image_path, e))
        return failure_image

    if size:
        image = image.resize(size)

    image = np.array(image)

    if not allow_non_rgb:
        if len(image.shape) != 3 or \
           (len(image.shape) == 3 and image.shape[-1] != 3):
            logging.warning('image at "{}" isn\'t RGB, therefore ' \
                            'not using it'.format(image_path))
            return failure_image

    return image
