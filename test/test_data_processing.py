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


# class Testread_gold_standard(unittest.TestCase):
#
#     actual_chinese_content = u'\n\n            <h>\u9ad8\u8003\u8bed\u6587\u5168\u7a0b\u68c0\u6d4b\u4e09\uff1a\u6b63\u786e\u4f7f\u7528\u8bcd\u8bed\uff08\u719f\u8bed\u4e00\uff09\n\n\n            <h>LEARNING.SOHU.COM    2004\u5e745\u670822\u65e515:36   \n\n\n   '
#
#     def test_read_gold_standard(self):
#         all_files = data_processing.get_list_all_corrected_files(FIXTURES)
#         all_files.sort()
#
#         chars = {'ascii': u'ascii yo!', 'iso-8859-1': u'\xd3', 'utf-8': u'\xae', 'utf-16': u'\xae'}
#
#         for e in chars:
#             content_comments = data_processing.read_gold_standard(
#                 FIXTURES, e)
#             actual_content = (u"Content here\nmore content\n" + chars[e] + u"\n")
#             self.assertEqual(content_comments[0], actual_content)
#             self.assertEqual(content_comments[1], '\nsome comments\n')
#
#     def test_utf8(self):
#         gs = ' '.join(data_processing.read_gold_standard(FIXTURES,
#             'utf-8_chinese'))
#         self.assertEqual(gs, Testread_gold_standard.actual_chinese_content)
#
#
# class Testextract_gold_standard(unittest.TestCase):
#
#     datadir = FIXTURES
#
#     @staticmethod
#     def _remove_output(names, catch_errors, name_maker=None):
#         import os
#         for fn in names:
#             if name_maker:
#                 fn_to_remove = name_maker(fn)
#             else:
#                 fn_to_remove = fn
#
#             if catch_errors:
#                 try:
#                     os.remove(fn_to_remove)
#                 except:
#                     pass
#             else:
#                 os.remove(fn_to_remove)
#
#     def test_extract_gold_standard(self):
#         test_files = ['page_comments', 'page_no_comments']
#         name_maker = lambda x: "%s/block_corrected/%s.block_corrected.txt" % (Testextract_gold_standard.datadir, x)
#
#         Testextract_gold_standard._remove_output(test_files, True, name_maker)
#
#         for f in test_files:
#             data_processing.extract_gold_standard(Testextract_gold_standard.datadir, f)
#             # check output file
#             with io.open(name_maker(f), mode='r', encoding='utf-8') as fout:
#                 corrected_blocks = fout.read()
#
#             with io.open(name_maker(f + '_expected'), mode='r', encoding='utf-8') as fexpected:
#                 expected_blocks = fexpected.read()
#
#             self.assertEqual(expected_blocks, corrected_blocks)
#
#         Testextract_gold_standard._remove_output(test_files, False, name_maker)
#
#
# if __name__ == "__main__":
#     unittest.main()
