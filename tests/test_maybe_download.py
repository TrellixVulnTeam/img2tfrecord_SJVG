#
# test_maybe_download
# img-tfrecord
#

"""

"""
from tests import SuperTestCase, TEST_OUTPUT_DIR
from imgNet2tfrecord.build_data import maybe_download


class TestMaybe_download(SuperTestCase):
    def test_maybe_download(self):

        # Check that build directory is created
        # Invalid CSV path
        self.assertRaises(AssertionError, maybe_download, TEST_OUTPUT_DIR, 'this.csv', 'user', 'pass')
