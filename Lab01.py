import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn import metrics


def clusteringByKmeans(Data, num_clusters):
    plt.figure()
    plt.scatter(Data[:, 0], Data[:, 1], marker="o", facecolors="none",
                edgecolors="black", s=80)
    x_min, x_max = Data[:, 0].min() - 1, Data[:, 0].max() + 1
    y_min, y_max = Data[:, 1].min() - 1, Data[:, 1].max() + 1
    plt.title("Вхідні Дані")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10)

    kmeans.fit(Data)

    step_size = 0.01

    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                                 np.arange(y_min, y_max, step_size))

    output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.clf()
    plt.imshow(output, interpolation="nearest",
               extent=(x_vals.min(), x_vals.max(),
                       y_vals.min(), y_vals.max()),
               cmap=plt.cm.Paired,
               aspect="auto",
               origin="lower")

    plt.scatter(Data[:, 0], Data[:, 1], marker="o", facecolors="none",
                edgecolors="black", s=80)

    cluster_centers = kmeans.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                marker="o", s=210, linewidths=4, color="black",
                zorder=12, facecolors="black")

    plt.title("Кордони кластерів")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def countOfClustersByUsingMeanShift(file, qntl):
    bandwidth_file = estimate_bandwidth(file, quantile=qntl, n_samples=len(file))

    meanshift_model = MeanShift(bandwidth=bandwidth_file, bin_seeding=True)
    meanshift_model.fit(file)

    cluster_centers = meanshift_model.cluster_centers_
    print("\nКоординати центрів кластерів:\n", cluster_centers)

    labels = meanshift_model.labels_
    num_clusters = len(np.unique(labels))
    print("\nКількість кластерів у вхідних даних= ", num_clusters)

    plt.figure()
    markers = "o*xvsd+.Xp"
    for i, marker in zip(range(num_clusters), markers):
        plt.scatter(file[labels == i, 0], file[labels == i, 1], marker=marker,
                    color="black")

        cluster_center = cluster_centers[i]
        plt.plot(cluster_center[0], cluster_center[1], marker="o",
                 markerfacecolor="red", markeredgecolor="black",
                 markersize=15)

    plt.title("Кластери")
    plt.show()


def getQualityOfClustering(file):
    scores = []
    values = np.arange(2, 15)

    for num_clusters in values:
        kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=10)

        kmeans.fit(file)

        score = metrics.silhouette_score(file, kmeans.labels_,
                                         metric="euclidean",
                                         sample_size=len(file))
        print("\nКількість кластерів = ", num_clusters)
        print("Силуетна оцінка = ", score)
        scores.append(score)

    plt.figure()
    plt.bar(values, scores, width=0.7, color="black", align="center")
    plt.title("Залежність силуетної оцінки від кількості кластерів")

    num_clusters = np.argmax(scores) + values[0]
    print("\nОптимальна кількість кластерів = ", num_clusters)

    plt.figure()
    plt.scatter(file[:, 0], file[:, 1],
                color="black",
                s=80,
                marker="o",
                facecolor="none")

    x_min, x_max = file[:, 0].min() - 1, file[:, 0].max() + 1
    y_min, y_max = file[:, 1].min() - 1, file[:, 1].max() + 1

    plt.title("Вхідні дані")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()
    return num_clusters


# data = np.loadtxt("data_clustering.txt", delimiter=",")
# quantile = 0.1

data = np.array(pd.read_csv("lab01.csv", delimiter=";"))
quantile = 0.14

# data = np.loadtxt("data_quality.txt", delimiter=",")


if __name__ == "__main__":
    countOfClustersByUsingMeanShift(data, quantile)
    num_clusters = getQualityOfClustering(data)
    clusteringByKmeans(data, num_clusters)
