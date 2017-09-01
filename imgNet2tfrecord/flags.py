#
# flags
# img-tfrecord
#

import tensorflow as tf

# Data and build paths
tf.app.flags.DEFINE_string(
    'raw_data_dir', '../data/raw_data',
    'Raw data directory, location where synset files and bounding boxes are downloaded to.')

tf.app.flags.DEFINE_string(
    'build_dir', '../data/data',
    'Output data directory, where the build TFRecord files are saved.')

tf.app.flags.DEFINE_string(
    'labels_dir', '../data/data/Annotation',
    'Labels root directory. Contains xml files for synset.')

# Utility file locations
tf.app.flags.DEFINE_string('download_list', '../etc/list.csv', 'Image-net synsets to download and use')
tf.app.flags.DEFINE_string('labels_file', '../etc/synset.txt', 'Labels file')
tf.app.flags.DEFINE_string('imagenet_metadata_file', '../etc/imagenet_metadata.txt', 'ImageNet metadata file')
tf.app.flags.DEFINE_string('bounding_box_file', '../etc/bounding_boxes.csv', 'Bounding boxes file.')

# Build configurations
tf.app.flags.DEFINE_boolean('download_data', False, 'Should download raw data.')
tf.app.flags.DEFINE_integer('train_shards', 90, 'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 10, 'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')

# ImageNet credentials
tf.app.flags.DEFINE_string('user', 'eskil', 'ImageNet username')
tf.app.flags.DEFINE_string(
    'access_pass', 'ed5ac67a10f56ba081f6735886b92d39959692e1',
    'ImageNet access pass. (Keep secret - for personal use only).')

FLAGS = tf.app.flags.FLAGS


def get_flags():
    return tf.flags.FLAGS
