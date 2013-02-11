
import unittest
from dragnet import util

class Test_evaluation_metrics(unittest.TestCase):
    def test_evaluation_metrics(self):

        predicted = 'skiing sparkling soft snow in soft sun'.split()
        actual = 'soft snow in soft sun soft turns turns'.split()


        def _f1(p, r):
            return 2 * p * r / (p + r)

        # for bag of words assumption
        p = 4.0 / 6.0
        r = 4.0 / 5
        f1 = _f1(p, r)

        prf = util.evaluation_metrics(predicted, actual)
        self.assertEqual((p, r, f1), prf)

        # for list assumption
        p = 5 / 7.0
        r = 5 / 8.0
        f1 = _f1(p, r)
        prf = util.evaluation_metrics(predicted, actual, bow=False)
        self.assertEqual((p, r, f1), prf)



if __name__ == "__main__":
    unittest.main()

