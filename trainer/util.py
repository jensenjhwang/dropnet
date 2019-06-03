import tensorflow as tf
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score

def find_best_clustering(input_data, clusters=None, train_steps=30):
    """
    Finds best k-means with number of clusters ranging between
    min_clusters(inclusive) and max_clusters(exclusive) for epochs

    input should be of shape (number of points, dimensions)
    """

    best_score = None
    best_model = None
    best_K = None
    tf.logging.set_verbosity('ERROR')

    def input_fn():
        return tf.data.Dataset.from_tensors(tf.convert_to_tensor(input_data, dtype=tf.float32))

    # for i in range(min_clusters, max_clusters):
    #     kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=i, use_mini_batch=False)
    #     for _ in range(train_steps):
    #         kmeans.train(input_fn)
    #     score = kmeans.score(input_fn)
    #     if best_score is None or score < best_score:
    #         best_score = score
    #         best_model = kmeans
    
    ### Init number of clusters
    # if clusters:
    #     num_clusters = clusters
    # else:
    #     num_clusters = int(np.sqrt(input.shape[0] / 2))

    ### Training (x1)
    # kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False)
    #     for _ in range(train_steps):
    #         kmeans.train(input_fn)
    #     score = kmeans.score(input_fn)
    #     if best_score is None or score < best_score:
    #         best_score = score
    #         best_model = kmeans

    min_clusters = 5
    max_clusters = 100
    scoring = "CHI"

    for K in range(min_clusters, max_clusters):
        kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=K, use_mini_batch=False)
        for _ in range(train_steps):
            kmeans.train(input_fn)

        if scoring == "BIC":
            score = compute_bic(input_data, kmeans, input_fn)
            print("K=",K," Bayesian Information Criteria:",score)
        elif scoring == "CSV":
            score = cl_size_var(input_data,kmeans,input_fn)
            print("K=",K," Cluster Size Variance:",score)
        elif scoring == "CSS":
            score = cl_size_var(input_data,kmeans,input_fn) * K
            print("K=",K," Cluster Size Square-sum:",score)
        elif scoring == "CHI":
            # https://stats.stackexchange.com/questions/21807/evaluation-measures-of-goodness-or-validity-of-clustering-without-having-truth/358937#358937
            labels = np.asarray(list(kmeans.predict_cluster_index(input_fn)))
            score = calinski_harabasz_score(input_data, labels)
            print("K=",K," C-H Index:",score)
        
        if best_score is None or score < best_score:
            best_K = K
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

    return best_score, list(best_model.predict_cluster_index(input_fn)), best_K

def compute_bic(input_data, kmeans,input_fn):
    """
    Computes the BIC metric for a given clusters.
    from: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
    adapted using example code from https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/KMeansClustering#cluster_centers

    Parameters:
    -----------------------------------------
    input_data: shape (number of points, dimensions)

    kmeans:  tf.contrib.factorization.KMeansClustering

    input_fn: from find_best_clustering

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = kmeans.cluster_centers()
    labels = np.asarray(list(kmeans.predict_cluster_index(input_fn)))
    print("labels:", labels.shape)
    # size of the clusters
    n = np.bincount(labels)
    #number of clusters
    m = len(n)
    print("number of clusters:",m)
    #size of data set
    N, d = input_data.shape
    print("N:",N,"d:",d)

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * kmeans.score(input_fn)

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

def cl_size_var(input_data, kmeans, input_fn):
    labels = np.asarray(list(kmeans.predict_cluster_index(input_fn)))
    n = np.bincount(labels)
    return np.var(n)


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
