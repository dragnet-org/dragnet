
from dragnet.lcs import check_inclusion
import unittest

class Test_check_inclusion(unittest.TestCase):
    def test_inclusion(self):
        inc = check_inclusion(["some", "words", "here", "the", "football"],
                        ["he", "said", "words", "kick", "the", "football"])
        self.assertTrue(inc, [False, True, False, True, True])


if __name__ == "__main__":
    unittest.main()


