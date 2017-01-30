import io
import os
from shutil import rmtree
import tempfile
import unittest

from dragnet import data_processing

FIXTURES = os.path.join('test', 'datafiles')


class TestGetFilenames(unittest.TestCase):

    def setUp(self):
        self.fileroots = ['bbc.co.story', 'f1', 'sad8-2sdkfj']
        self.data_dir = tempfile.mkdtemp()
        for froot in self.fileroots:
            fname = os.path.join(self.data_dir, '{}.html.corrected.txt'.format(froot))
            with io.open(fname, mode='wt') as f:
                f.write(u'.')

    def test_get_filenames(self):
        filenames = list(data_processing.get_filenames(self.data_dir))
        self.assertEqual(
            filenames,
            ['{}.html.corrected.txt'.format(froot) for froot in self.fileroots]
            )

    def test_get_filenames_full_path(self):
        filenames = list(data_processing.get_filenames(
            self.data_dir, full_path=True))
        self.assertEqual(
            filenames,
            [os.path.join(self.data_dir, '{}.html.corrected.txt'.format(froot))
             for froot in self.fileroots]
            )

    def test_get_filenames_match_regex(self):
        filenames = list(data_processing.get_filenames(
            self.data_dir, match_regex='f1'))
        self.assertEqual(filenames, ['f1.html.corrected.txt'])
        filenames = list(data_processing.get_filenames(
            self.data_dir, match_regex='foo'))
        self.assertEqual(filenames, [])

    def test_get_filenames_extension(self):
        filenames = list(data_processing.get_filenames(
            self.data_dir, extension='.txt'))
        self.assertEqual(
            filenames,
            ['{}.html.corrected.txt'.format(froot) for froot in self.fileroots]
            )
        filenames = list(data_processing.get_filenames(
            self.data_dir, extension='.foo'))
        self.assertEqual(filenames, [])

    def tearDown(self):
        rmtree(self.data_dir)


class TestReadGoldStandard(unittest.TestCase):

    actual_chinese_content = u'<h>\u9ad8\u8003\u8bed\u6587\u5168\u7a0b\u68c0\u6d4b\u4e09:\u6b63\u786e\u4f7f\u7528\u8bcd\u8bed(\u719f\u8bed\u4e00)\n\n\n            <h>LEARNING.SOHU.COM    2004\u5e745\u670822\u65e515:36 '

    def test_read_gold_standard(self):
        tests = {'ascii': u'ascii yo!',
                 'iso-8859-1': u'\xd3',
                 'utf-8': u'\xae',
                 'utf-16': u'\xae'}
        for encoding, expected in tests.items():
            content_comments = data_processing.read_gold_standard_file(
                FIXTURES, encoding)
            self.assertEqual(content_comments[0], u'Content here\nmore content\n' + expected)
            self.assertEqual(content_comments[1], 'some comments')

    def test_utf8_chinese(self):
        gs = ' '.join(data_processing.read_gold_standard_file(FIXTURES, 'utf-8_chinese'))
        self.assertEqual(gs, self.actual_chinese_content)


class TestExtractGoldStandard(unittest.TestCase):

    def test_extract_gold_standard(self):
        make_filepath = lambda x: os.path.join(FIXTURES, 'block_corrected', '{}.block_corrected.txt'.format(x))

        fileroots = ['page_comments', 'page_no_comments']
        for fileroot in fileroots:
            actual_filepath = make_filepath(fileroot)
            expected_filepath = make_filepath(fileroot + '_expected')
            data_processing.extract_gold_standard(FIXTURES, fileroot)

            with io.open(actual_filepath, mode='rt') as f:
                actual_blocks = f.read()
            with io.open(expected_filepath, mode='rt') as f:
                expected_blocks = f.read()

            os.remove(actual_filepath)
            self.assertEqual(expected_blocks, actual_blocks)


if __name__ == "__main__":
    unittest.main()
