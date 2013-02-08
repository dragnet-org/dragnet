
import unittest
from dragnet import data_processing
import codecs

class Testget_list_all_corrected_files(unittest.TestCase):
    def test_list_all_files(self):
        all_files = data_processing.get_list_all_corrected_files("datafiles")
        all_files.sort()
        self.assertEqual(all_files,
        [('datafiles/Corrected/ascii.html.corrected.txt', 'ascii'),
         ('datafiles/Corrected/iso-8859-1.html.corrected.txt', 'iso-8859-1'),
         ('datafiles/Corrected/page_comments.html.corrected.txt', 'page_comments'),
         ('datafiles/Corrected/page_no_comments.html.corrected.txt',
          'page_no_comments'),
         ('datafiles/Corrected/utf-16.html.corrected.txt', 'utf-16'),
         ('datafiles/Corrected/utf-8.html.corrected.txt', 'utf-8')])



class Testread_gold_standard(unittest.TestCase):
    def test_read_gold_standard(self):
        all_files = data_processing.get_list_all_corrected_files("datafiles")
        all_files.sort()

        chars = {'ascii':u'ascii yo!', 'iso-8859-1':u'\x0401', 'utf-8':u'\xae', 'utf-16':u'\xae'}

        for fname, e in all_files:
            if e in chars:
                content_comments = data_processing.read_gold_standard("datafiles", e)
                actual_content = (u"Content here\nmore content\n" +
                    chars[e] + u"\n").encode('utf-8')
                self.assertEqual(content_comments[0], actual_content)
                self.assertEqual(content_comments[1], '\nsome comments\n')


class Testextract_gold_standard(unittest.TestCase):

    datadir = "datafiles"

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


