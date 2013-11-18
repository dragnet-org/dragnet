
import unittest
from dragnet import kmeans
import numpy as np

class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[4, 5.0], [1.0, 2.0], [8.1, 2.6], [5.5, 6.5], [1.2, 2.2]])


    def test_update_centers(self):
        km = kmeans.KMeans(3)
        km.fit(self.X)
        km._init_centers(self.X)
        km.update_centers(self.X, np.array([0, 1, 1, 0, 2]))
        centers_actual = np.array( [[0.5 * (4 + 5.5), 0.5 * (5 + 6.5)],
                  [0.5 * (1.0 + 8.1), 0.5 * (2.0 + 2.6)],
                  [1.2, 2.2]])
        self.assertTrue(np.allclose(centers_actual, km.centers))

if __name__ == "__main__":
    unittest.main()

