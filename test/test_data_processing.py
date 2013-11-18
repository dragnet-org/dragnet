
import unittest
from dragnet import data_processing
import codecs

import tempfile
import os

from shutil import rmtree

FIXTURES = 'test/datafiles'

class Testget_list_all_corrected_files(unittest.TestCase):
    def test_list_all_files(self):
        prefixes = ['bbc.co.story', 'f1', 'sad8-2sdkfj']
        datadir = tempfile.mkdtemp()
        os.mkdir(datadir + '/Corrected')
        for froot in prefixes:
            with open(datadir + '/Corrected/%s.html.corrected.txt' % froot, 'w') as f:
                f.write('.')

        all_files = data_processing.get_list_all_corrected_files(datadir)
        all_files.sort()
        self.assertEqual(all_files,
            [('%s/Corrected/%s.html.corrected.txt' %
                (datadir, froot), froot)
              for froot in prefixes])

        rmtree(datadir)


class Testread_gold_standard(unittest.TestCase):

    actual_chinese_content = u'\n\n            <h>\u9ad8\u8003\u8bed\u6587\u5168\u7a0b\u68c0\u6d4b\u4e09\uff1a\u6b63\u786e\u4f7f\u7528\u8bcd\u8bed\uff08\u719f\u8bed\u4e00\uff09\n\n\n            <h>LEARNING.SOHU.COM    2004\u5e745\u670822\u65e515:36   \n\n\n   '

    def test_read_gold_standard(self):
        all_files = data_processing.get_list_all_corrected_files(FIXTURES)
        all_files.sort()

        chars = {'ascii':u'ascii yo!', 'iso-8859-1':u'\xd3', 'utf-8':u'\xae', 'utf-16':u'\xae'}

        for e in chars:
            content_comments = data_processing.read_gold_standard(
                FIXTURES, e)
            actual_content = (u"Content here\nmore content\n" + chars[e] + u"\n")
            self.assertEqual(content_comments[0], actual_content)
            self.assertEqual(content_comments[1], '\nsome comments\n')

    def test_utf8(self):
        gs = ' '.join(data_processing.read_gold_standard(FIXTURES,
            'utf-8_chinese'))
        self.assertEqual(gs, Testread_gold_standard.actual_chinese_content)




class Testextract_gold_standard(unittest.TestCase):

    datadir = FIXTURES

    @staticmethod
    def _remove_output(names, catch_errors, name_maker=None):
        import os
        for fn in names:
            if name_maker:
                fn_to_remove = name_maker(fn)
            else:
                fn_to_remove = fn

            if catch_errors:
                try:
                    os.remove(fn_to_remove)
                except:
                    pass
            else:
                os.remove(fn_to_remove)


    def test_extract_gold_standard(self):
        test_files = ['page_comments', 'page_no_comments']
        name_maker = lambda x : "%s/block_corrected/%s.block_corrected.txt" % (Testextract_gold_standard.datadir, x)

        Testextract_gold_standard._remove_output(test_files, True, name_maker)

        for f in test_files:
            data_processing.extract_gold_standard(Testextract_gold_standard.datadir, f)
            # check output file
            with codecs.open(name_maker(f), 'r', encoding='utf-8') as fout:
                corrected_blocks = fout.read()

            with codecs.open(name_maker(f + '_expected'), 'r', encoding='utf-8') as fexpected:
                expected_blocks = fexpected.read()

            self.assertEqual(expected_blocks, corrected_blocks)

        Testextract_gold_standard._remove_output(test_files, False, name_maker)


if __name__ == "__main__":
    unittest.main()


