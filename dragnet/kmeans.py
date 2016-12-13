import numpy as np


class KMeans(object):

    def __init__(self, clusters):
        self.clusters = clusters
        self.centers = None  # (nclusters X nx)
        self._nx = None
        self._errtol = 1e-3
        self._maxiterations = 50

    def closest_centers(self, X):
        """Returns the indices of the closest clusters to X
        Return is length(X.shape[0])"""
        distances = np.zeros((X.shape[0], self.clusters))
        for k in xrange(self.clusters):
            distances[:, k] = np.sum((X - self.centers[k, :]) ** 2, axis=1)
        return distances.argmin(axis=1)

    def update_centers(self, X, cc):
        for k in xrange(self.clusters):
            these_indices = cc == k
            if np.any(these_indices):
                self.centers[k, :] = np.mean(X[these_indices, :], axis=0)
            else:
                # cluster is empty.  pick a randon one
                self._init_one_center(X, k)

    def _init_one_center(self, X, k):
        # initialize center k randomly
        self.centers[k, :] = X[np.random.randint(X.shape[0]), :]

    def _init_centers(self, X):
        """initialize centers to random values"""
        # randomly assign data in X to a cluster center, then take means
        self.centers = np.zeros((self.clusters, self._nx))
        for k in xrange(self.clusters):
            self._init_one_center(X, k)

    def fit(self, X):
        """X = nobs, nfeatures) array"""
        self._nx = X.shape[1]
        self._init_centers(X)

        last_centers = self.centers.copy()
        converged = False
        iteration = 1
        while not converged and iteration < self._maxiterations:
            iteration += 1
            # (1) find closest centers
            indices = self.closest_centers(X)

            # (2) update centers
            self.update_centers(X, indices)

            # (3) check for convergence
            converged = np.abs(last_centers - self.centers).max() < self._errtol
            last_centers = self.centers.copy()

    def plot_clusters(self, X):
        assert X.shape[1] == 2
        fig = plt.figure(1)
        fig.clf()

        colors = ['b', 'r', 'g']
        indices = self.closest_centers(X)
        for k in xrange(self.clusters):
            this_cluster = indices == k
            plt.scatter(X[this_cluster, 0], X[this_cluster, 1], s=2, color=colors[k])
            plt.plot(self.centers[k][0], self.centers[k][1], 'kx', markersize=10)

        fig.show()


class KMeansFixedOrigin(KMeans):
    """KMeans with once cluster always fixed to origin"""

    def _init_centers(self, X):
        super(KMeansFixedOrigin, self)._init_centers(X)
        self.centers[0, :] = np.zeros(self._nx)

    def update_centers(self, X, cc):
        super(KMeansFixedOrigin, self).update_centers(X, cc)
        self.centers[0, :] = np.zeros(self._nx)
