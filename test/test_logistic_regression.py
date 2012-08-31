
import unittest
from dragnet import LogisticRegression
import numpy as np

class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[1, -2], [-0.5, -2]])
        self.t = np.array([0, 1])
        self.w = np.array([3, -1])
        self.b = 1
        self.lam = 7

    def test_sigmoid(self):
        y = LogisticRegression._sigmoid(self.x, self.b, self.w)
        yact = np.array([1.0 / (1.0 + np.exp(-6)), 1.0 / (1.0 + np.exp(-1.5))])

        self.assertTrue(np.all(np.abs(y - yact) < 1.0e-12))

    def test_error(self):
        x0 = np.array([self.x[0]])
        x1 =  np.array([self.x[1]])
        error = LogisticRegression._loss(x0, x1, self.b, self.w, self.lam)

        # this assumes test_sigmoid pases
        err_act = -np.log(LogisticRegression._sigmoid(x1, self.b, self.w)) - np.log(1.0 - LogisticRegression._sigmoid(x0, self.b, self.w)) + 0.5 * 7 * 10
        self.assertTrue( abs(float(err_act) - error) < 1.0e-12 ) 

        # weighted case
        x00 = np.array([self.x[0], [55, -2]])
        error_weighted = LogisticRegression._loss(x00, x1, self.b, self.w, self.lam, [np.array([0.4, 0.75]), np.array(0.35)])
        err_weighted_act = -np.log(LogisticRegression._sigmoid(x1, self.b, self.w)) * 0.35 - np.log(1.0 - LogisticRegression._sigmoid(x0, self.b, self.w)) * 0.4 - np.log(1.0 - LogisticRegression._sigmoid([x00[1, :]], self.b, self.w)) * 0.75 + 0.5 * 7 * 10
        self.assertTrue( abs(float(err_weighted_act) - error_weighted) < 1.0e-12 )

    def test_gradient(self):
        gradient = LogisticRegression._gradient_loss(self.x, self.t, self.b, self.w, self.lam)
        gradient_act = np.array([0.0, 7 * 3, 7 * -1])
        error = LogisticRegression._sigmoid(self.x, self.b, self.w) - self.t
        gradient_act[0] = np.sum(error)
        gradient_act[1] += np.sum(error * self.x[:, 0])
        gradient_act[2] += np.sum(error * self.x[:, 1])

        self.assertTrue(np.all(np.abs(gradient - gradient_act) < 1.0e-12))


if __name__ == "__main__":
    unittest.main()


