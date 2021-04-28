import faiss
import numpy as np


class FaissKMeans(object):
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(
            d=X.shape[1],
            k=self.n_clusters,
            niter=self.max_iter,
            nredo=self.n_init
        )
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

        # print("Centers:")
        # for cen in self.cluster_centers_:
        #     print(cen)

    def get_centers(self):
        return self.cluster_centers_

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]



if __name__ == "__main__":
    KMeans = FaissKMeans(n_clusters=2)
    X = np.array(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [0, 1, 2],
         [3, 4, 5],
         [6, 7, 8],
         [9, 0, 1]
         ]
    )

    KMeans.fit(X)
    predicts = KMeans.predict(np.array([[5, 5, 5], [3, 3, 3]])).squeeze()
    centers = KMeans.get_centers()
    print(centers[predicts])