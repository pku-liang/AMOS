import faiss
import os
import json
import numpy as np


class FaissKMeans(object):
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, n_init=10, max_iter=300):
        self.kmeans = faiss.Kmeans(
            d=X.shape[1],
            k=self.n_clusters,
            niter=max_iter,
            nredo=n_init
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

    def topk(self, X, k):
        return self.kmeans.index.search(X.astype(np.float32), k)[1]

    def save(self, path, override=True):
        obj = {"centers": self.cluster_centers_.tolist()}
        s = json.dumps(obj)
        if not override:
            assert not os.path.isfile(path)
        with open(path, "w") as fout:
            fout.write(s + "\n")

    @staticmethod
    def load(path):
        with open(path, "r") as fin:
            line = fin.readline().strip()
            obj = json.loads(line)
            centers = np.array(obj["centers"])
            n_clusters = len(centers)
            ret = FaissKMeans(n_clusters=n_clusters)
            ret.fit(centers)
            return ret




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
    print(predicts)
    centers = KMeans.get_centers()
    print(centers[predicts])

    KMeans.save("tmp.json")
    new_kmeans = FaissKMeans.load("tmp.json")
    predicts = new_kmeans.predict(np.array([[5, 5, 5], [3, 3, 3]])).squeeze()
    print(predicts)