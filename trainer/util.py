import tensorflow as tf

def find_best_clustering(input, min_clusters, max_clusters, epochs):
    """
    Finds best k-means with number of clusters ranging between
    min_clusters(inclusive) and max_clusters(exclusive) for epochs

    input should be of shape (number of points, dimensions)
    """
    def input_fn():
        return tf.data.Dataset.from_tensors(input).repeat(num_epochs)

    for i in range(min_clusters, max_clusters):
        kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=i, use_mini_batch=False)
