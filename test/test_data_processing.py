import io
import os
from shutil import rmtree
import tempfile

import pytest

from dragnet import data_processing

FIXTURES = os.path.join('test', 'datafiles')


@pytest.fixture(scope="module")
def fileroots():
    return ["bbc.co.story", "f1", "sad8-2sdkfj"]


@pytest.fixture(scope="class")
def datadir(fileroots):
    datadir = tempfile.mkdtemp()
    for froot in fileroots:
        fname = os.path.join(datadir, "{}.html.corrected.txt".format(froot))
        with io.open(fname, mode="wt") as f:
            f.write(u".")
    yield datadir
    rmtree(datadir)


@pytest.mark.usefixtures("datadir")
class TestGetFilenames(object):

    def test_get_filenames(self, fileroots, datadir):
        filenames = list(data_processing.get_filenames(datadir))
        assert (
            filenames ==
            ["{}.html.corrected.txt".format(froot) for froot in fileroots]
        )

    def test_get_filenames_full_path(self, fileroots, datadir):
        filenames = list(data_processing.get_filenames(datadir, full_path=True))
        assert (
            filenames ==
            [os.path.join(datadir, "{}.html.corrected.txt".format(froot))
             for froot in fileroots]
        )

    def test_get_filenames_match_regex(self, datadir):
        filenames = list(data_processing.get_filenames(datadir, match_regex='f1'))
        assert filenames == ['f1.html.corrected.txt']
        filenames = list(data_processing.get_filenames(datadir, match_regex='foo'))
        assert filenames == []

    def test_get_filenames_extension(self, fileroots, datadir):
        filenames = list(data_processing.get_filenames(datadir, extension='.txt'))
        assert (
            filenames ==
            ['{}.html.corrected.txt'.format(froot) for froot in fileroots]
        )
        filenames = list(data_processing.get_filenames(datadir, extension='.foo'))
        assert filenames == []


class TestReadGoldStandard(object):

    def test_read_gold_standard(self):
        tests = {
            'ascii': u'ascii yo!',
            'iso-8859-1': u'\xd3',
            'utf-8': u'\xae',
            'utf-16': u'\xae',
        }
        for encoding, expected in tests.items():
            content_comments = data_processing.read_gold_standard_file(
                FIXTURES, encoding)
            assert content_comments[0] == u"Content here\nmore content\n" + expected
            assert content_comments[1] == "some comments"

    def test_utf8_chinese(self):
        actual_chinese_content = u'<h>\u9ad8\u8003\u8bed\u6587\u5168\u7a0b\u68c0\u6d4b\u4e09\uff1a\u6b63\u786e\u4f7f\u7528\u8bcd\u8bed\uff08\u719f\u8bed\u4e00\uff09\n\n\n            <h>LEARNING.SOHU.COM    2004\u5e745\u670822\u65e515:36 '
        gs = " ".join(data_processing.read_gold_standard_file(FIXTURES, "utf-8_chinese"))
        assert gs == actual_chinese_content


def make_filepath(s):
    return os.path.join(FIXTURES, "block_corrected", "{}.block_corrected.txt".format(s))


class TestExtractGoldStandard(object):

    def test_extract_gold_standard(self):
        fileroots = ["page_comments", "page_no_comments"]
        for fileroot in fileroots:
            actual_filepath = make_filepath(fileroot)
            expected_filepath = make_filepath(fileroot + "_expected")
            data_processing.extract_gold_standard_blocks(FIXTURES, fileroot)
            with io.open(actual_filepath, mode="rt") as f:
                actual_blocks = f.read()
            with io.open(expected_filepath, mode="rt") as f:
                expected_blocks = f.read()
            os.remove(actual_filepath)
            assert expected_blocks == actual_blocks

    def test_extract_blank_label(self):
        blank_label = data_processing.read_gold_standard_blocks_file(FIXTURES, "blank_label")
        assert len(list(blank_label)) == 0
        blank_data = data_processing.prepare_data(FIXTURES, "blank_label")
        assert len(blank_data[0]) > 0
