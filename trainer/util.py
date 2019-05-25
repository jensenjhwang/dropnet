import tensorflow as tf
import numpy as np
from sklearn.metrics import adjusted_rand_score

def find_best_clustering(input, clusters=None, train_steps=30):
    """
    Finds best k-means with number of clusters ranging between
    min_clusters(inclusive) and max_clusters(exclusive) for epochs

    input should be of shape (number of points, dimensions)
    """

    best_score = None
    best_model = None
    tf.logging.set_verbosity('ERROR')

    def input_fn():
        return tf.data.Dataset.from_tensors(tf.convert_to_tensor(input, dtype=tf.float32))

    # for i in range(min_clusters, max_clusters):
    #     kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=i, use_mini_batch=False)
    #     for _ in range(train_steps):
    #         kmeans.train(input_fn)
    #     score = kmeans.score(input_fn)
    #     if best_score is None or score < best_score:
    #         best_score = score
    #         best_model = kmeans
    if clusters:
        num_clusters = clusters
    else:
        num_clusters = int(np.sqrt(input.shape[0] / 2))
    kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False)
    for _ in range(train_steps):
        kmeans.train(input_fn)
    score = kmeans.score(input_fn)
    if best_score is None or score < best_score:
        best_score = score
        best_model = kmeans

    # for i in range(min_clusters, max_clusters):
    #     g2 = tf.Graph()
    #     with g2.as_default() as g:
    #         copied = tf.identity(input)
    #         def input_fn():
    #             return tf.data.Dataset.from_tensors(copied).repeat(1)
    #         kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=i, use_mini_batch=False)
    #         for _ in range(train_steps):
    #           kmeans.train(input_fn)
    #         score = kmeans.score(input_fn)
    #         if best_score is None or score < best_score:
    #             best_score = score
    #             best_model = kmeans

    tf.logging.set_verbosity('DEBUG')

    return best_score, list(best_model.predict_cluster_index(input_fn))

def rand_index(cluster_1, cluster_2):

    score = 0
    length = len(cluster_1)
    for i in range(length):
        for j in range(i+1, length):
            if not ((cluster_1[i] == cluster_1[j]) ^ (cluster_2[i] == cluster_2[j])):
                score += 1

    return 2 * score / length  / (length - 1)

def adjusted_rand_index(cluster_1, cluster_2):

    return adjusted_rand_score(cluster_1, cluster_2)

def match_proportion(base_cluster, cluster):

    score = 0
    count = 0
    length = len(base_cluster)
    for i in range(length):
        for j in range(i+1, length):
            if (base_cluster[i] == base_cluster[j]):
                count += 1
                if (cluster[i] == cluster[j]):
                    score += 1

    return score / count
