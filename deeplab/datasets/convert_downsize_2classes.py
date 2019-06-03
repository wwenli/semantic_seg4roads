# author: wenli wang
# ==============================================================================

import math
import os
import random
import string
import sys
import build_data
import tensorflow as tf
from PIL import Image
import numpy

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_image_label_folder',
    './mapillary/mapillary_dataset/training/labels_512_384',
    'Folder containing annotations for trainng images')

tf.app.flags.DEFINE_string(
    'val_image_label_folder',
    './mapillary/mapillary_dataset/validation/labels_512_384',
    'Folder containing annotations for validation')

tf.app.flags.DEFINE_string(
    'train_image_label_lane_marking_general_folder',
    './mapillary/mapillary_dataset/training/labels_512_384_lane_marking_general',
    'Folder containing annotations for trainng images')

tf.app.flags.DEFINE_string(
    'val_image_label_lane_marking_general_folder',
    './mapillary/mapillary_dataset/validation/labels_512_384_lane_marking_general',
    'Folder containing annotations for validation')


def _convert_dataset(dataset_split, dataset_label_dir, dataset_new_label_dir):
    """ Converts the Mapillary dataset into into tfrecord format (SSTable).

    Args:
      dataset_split: Dataset split (e.g., train, val).
      dataset_label_dir: Dir in which the annotations locates.
      dataset_new_label_dir: new annotation locates
    """

    seg_names = tf.gfile.Glob(os.path.join(dataset_label_dir, '*.png'))
    num_images = len(seg_names)

    for k in range(num_images):
        sys.stdout.write('\r>> Converting image %d' % (k + 1))
        sys.stdout.flush()
        # Read the semantic segmentation annotation.
        seg_filename = seg_names[k]
        # Convert to two classes.
        im = Image.open(seg_filename, 'r')
        pixels = im.load()
        # print im.size  # Get the width and hight of the image for iterating over
        # print pixels[0, 0]  # Get the RGBA Value of the a pixel of an image

        img = Image.new('L', im.size)
        pixelsNew = img.load()

        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixels[i, j] == 24:
                    pixelsNew[i, j] = 1
                else:
                    pixelsNew[i, j] = 0

        img.save(os.path.join(dataset_new_label_dir, os.path.basename(seg_filename)))

    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
    tf.gfile.MakeDirs(FLAGS.train_image_label_lane_marking_general_folder)
    tf.gfile.MakeDirs(FLAGS.val_image_label_lane_marking_general_folder)
    _convert_dataset('training', FLAGS.train_image_label_folder, FLAGS.train_image_label_lane_marking_general_folder)
    _convert_dataset('validation', FLAGS.val_image_label_folder, FLAGS.val_image_label_lane_marking_general_folder)


if __name__ == '__main__':
    tf.app.run()
