
#
# __init__.py
# img-tfrecord.tests
#
from unittest import TestCase
import shutil

TEST_OUTPUT_DIR = 'test_build'


class SuperTestCase(TestCase):
    def setUp(self):
        print('init test')

    def tearDown(self):
        shutil.rmtree(TEST_OUTPUT_DIR)
