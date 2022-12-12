#
# build_data
# imgNet2tfrecord
#

"""
Main Script for building image data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import shutil
import sys
import tarfile

import numpy as np
import tensorflow as tf

from imgNet2tfrecord import image_util
from imgNet2tfrecord import process_bounding_boxes
from imgNet2tfrecord import flags

if sys.version_info >= (3,):
    from urllib import request as urllib2

FLAGS = flags.get_flags()

# URL CONSTANTS
SYNSET_URL = 'http://www.image-net.org/download/synset?wnid={}&username={}&accesskey={}&release=latest&src=stanford'
BBOX_URL = 'http://www.image-net.org/downloads/bbox/bbox/{}.tar.gz'


class UnpackInfo(object):
    def __init__(self, name, num_shards, output_dir=FLAGS.build_dir):
        self.name = name
        self.num_shards = num_shards
        self.output_dir = '{}/{}'.format(output_dir, name)
        self.labels = np.empty(0, dtype=int)
        self.filenames = np.empty(0)
        self.classnames = np.empty(0)
        self.human_classnames = np.empty(0)
        self.bboxes = np.empty(0)

    def add_data(self, filenames, label, classname):
        self.filenames = np.append(self.filenames, filenames)
        self.labels = np.append(self.labels, [label] * filenames.size)
        self.classnames = np.append(self.classnames, [classname] * filenames.size)

    def shuffle_data(self):
        shuffle_array = np.arange(self.labels.shape[0])
        np.random.shuffle(shuffle_array)

        self.filenames = self.filenames[shuffle_array]
        self.labels = self.labels[shuffle_array]
        self.classnames = self.classnames[shuffle_array]

    def set_human_classnames(self, synset_lookup):
        humans = []
        for s in self.classnames:
            assert s in synset_lookup, ('Failed to find: %s' % s)
            humans.append(synset_lookup[s])
        self.human_classnames = np.array(humans, dtype=str)

    def set_bboxes(self, bbox_lookup):
        num_image_bbox = 0
        bboxes = []
        for f in self.filenames:
            basename = os.path.basename(f)
            if basename in bbox_lookup:
                bboxes.append(bbox_lookup[basename])
                num_image_bbox += 1
            else:
                bboxes.append([])
        print('Found %d images with bboxes out of %d images' % (
            num_image_bbox, len(self.filenames)))
        self.bboxes = bboxes

    def __repr__(self):
        return 'Info: {self.name}'.format(self=self)


def get_dl_urls(wnid, user, access_pass):

    """
    Returns download url for images and bounding boxes
    :param str wnid: Wordnet id
    :return: Synset url and bbox url for the given wordnet id.
    """

    synset_url = SYNSET_URL.format(wnid, user, access_pass)
    bbox_url = BBOX_URL.format(wnid)
    return [synset_url, bbox_url]


def maybe_download(dest_directory, download_list, user, access_pass):
    """
    Download Image-net synsets and bounding boxes
    :param str dest_directory : Path to build directory.
    :param str download_list : Path to download list csv file.
    :param str user: ImageNet username
    :param str access_pass: ImageNet pass
    """
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    assert os.path.exists(download_list), 'Download CSV file does not exist.'

    # Parse CSV - download synsets and bounding boxes
    with open(download_list) as csvfile:
        list_reader = csv.reader(csvfile)
        for row in list_reader:

            wordnet_id = row[0]
            urls = get_dl_urls(wordnet_id, user, access_pass)
            filenames = ['{}.tar'.format(wordnet_id), '{}.tar.gz'.format(wordnet_id)]

            for i, filename in enumerate(filenames):
                filepath = os.path.join(dest_directory, filename)

                if not os.path.exists(filepath):
                    download_file(urls[i], filepath)

                # If last row is 0, don't download bounding boxes
                if row[2] == '0':
                    break


def download_file(url, filename):
    u = urllib2.urlopen(url)

    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)

            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)


def write_synsets_file(synsets):
    with open(FLAGS.labels_file, 'w') as f:
        f.write('\n'.join(synsets))


def unpack_data(data_dir, output_dir):
    """Unpack and split raw data into training and validation sets."""
    synset_labels = []
    label = 0

    validation_data = UnpackInfo(name='validation', num_shards=FLAGS.validation_shards)
    train_data = UnpackInfo(name='train', num_shards=FLAGS.train_shards)

    tar_format = '%s/*.tar' % data_dir
    data_files = tf.gfile.Glob(tar_format)

    print('Unpacking raw data FROM: {} TO: {}'.format(data_dir, output_dir))

    for file in data_files:
        filename = file.split('/')[-1].split('.tar')[0]
        assert isinstance(filename, str) and len(filename) > 0
        synset_labels.append(filename)
        label += 1

        with tarfile.open(file) as tar:
            path = '{}/{}'.format(output_dir, filename)
            names = np.array(tar.getnames(), dtype=str)
            # noinspection PyTypeChecker
            path_prefix = np.full_like(names, fill_value='/')
            names = np.core.defchararray.add(path_prefix, names)
            # noinspection PyTypeChecker
            file_prefix = np.full_like(names, fill_value=path)
            names = np.core.defchararray.add(file_prefix, names)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=path)

        # Extract bbox
        bbox_file = '{}.gz'.format(file)
        if os.path.isfile(bbox_file):
            with tarfile.open(bbox_file) as tar_gz:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar_gz, path=output_dir)

        train_names, valid_names = np.split(names, [int(0.9 * names.size)])

        train_data.add_data(train_names, label, filename)
        validation_data.add_data(valid_names, label, filename)

    train_data.shuffle_data()
    validation_data.shuffle_data()

    write_synsets_file(synset_labels)

    return train_data, validation_data, synset_labels


def clean_temp_data(output_dir, u_labels):
    # print(u_labels)
    for label in u_labels:
        shutil.rmtree('{}/{}'.format(output_dir, label))


def main(_):
    assert not FLAGS.train_shards % FLAGS.num_threads, \
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards'
    assert not FLAGS.validation_shards % FLAGS.num_threads, \
        'Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards'

    assert isinstance(FLAGS.user, str), 'Imagenet username must be provided.'
    assert isinstance(FLAGS.access_pass, str), 'Imagenet Access pass must be provided.'

    # Check if raw data should be downloaded
    maybe_download(FLAGS.raw_data_dir, FLAGS.download_list, FLAGS.user, FLAGS.access_pass)

    # Unpack raw data
    train, valid, labels = unpack_data(FLAGS.raw_data_dir, FLAGS.build_dir)

    # Process bounding boxes
    process_bounding_boxes.run(FLAGS.labels_dir, labels, FLAGS.bounding_box_file)

    # Build lookups
    synset_lookup, bbox_lookup = image_util.build_lookups(FLAGS.imagenet_metadata_file, FLAGS.bounding_box_file,
                                                          labels)
    train.set_human_classnames(synset_lookup)
    train.set_bboxes(bbox_lookup)
    valid.set_human_classnames(synset_lookup)
    valid.set_bboxes(bbox_lookup)

    # # Process images and write TF Records (shards)
    image_util.process_image_files(train, FLAGS.num_threads)
    image_util.process_image_files(valid, FLAGS.num_threads)
    #
    # # Clean temp JPEG files
    clean_temp_data(FLAGS.build_dir, labels)


if __name__ == '__main__':
    tf.app.run()
