
import unittest
from dragnet import data_processing

class Testget_list_all_corrected_files(unittest.TestCase):
    def test_list_all_files(self):
        all_files = data_processing.get_list_all_corrected_files("datafiles")
        all_files.sort()
        self.assertEqual(all_files,
                [('datafiles/Corrected/ascii.html.corrected.txt', 'ascii'),
                 ('datafiles/Corrected/iso-8859-1.html.corrected.txt', 'iso-8859-1'),
                 ('datafiles/Corrected/utf-16.html.corrected.txt', 'utf-16'),
                 ('datafiles/Corrected/utf-8.html.corrected.txt', 'utf-8')])


class Testread_gold_standard(unittest.TestCase):
    def test_read_gold_standard(self):
        all_files = data_processing.get_list_all_corrected_files("datafiles")
        all_files.sort()

        chars = {'ascii':u'ascii yo!', 'iso-8859-1':u'\x0401', 'utf-8':u'\xae', 'utf-16':u'\xae'}

        for fname, e in all_files:
            content_comments = data_processing.read_gold_standard("datafiles", e)
            actual_content = (u"Content here\nmore content\n" +
                chars[e] + u"\n").encode('utf-8')
            self.assertEqual(content_comments[0], actual_content)
            self.assertEqual(content_comments[1], '\nsome comments\n')


if __name__ == "__main__":
    unittest.main()


