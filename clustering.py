import os
import pickle
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import pandas as pd

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)


def find_most_frequent_elements(input_list):
    """
    Find the elements that appear most frequently in the given list.

    Parameters:
    - input_list: list of elements to analyze.

    Returns:
    - A tuple containing a list of the most frequent elements and their frequency.
    """
    from collections import Counter

    # Using Counter to find frequencies and identify the most common element(s)
    frequency = Counter(input_list)
    most_common = frequency.most_common()  # Sorted list of tuples (element, frequency)

    # Extract the highest frequency
    highest_frequency = most_common[0][1]

    # Extract all elements with the highest frequency
    most_frequent_elements = [element for element, freq in most_common if freq == highest_frequency]

    return most_frequent_elements, highest_frequency


def dbscan_param_search(input_data):
    # Defining the list of hyperparameters to try
    eps_list = np.arange(start=0.1, stop=5, step=0.1)
    min_sample_list = np.arange(start=2, stop=25, step=1)
    score = -1
    best_dbscan = DBSCAN()
    # Creating empty data frame to store the silhouette scores for each trials
    silhouette_scores_data = pd.DataFrame()
    for eps_trial in eps_list:
        for min_sample_trial in min_sample_list:
            # Generating DBSAN clusters
            db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial).fit(input_data)
            labels = db.labels_
            # print(labels)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                if score <= silhouette_score(input_data, labels):
                    score = silhouette_score(input_data, labels)
                    best_dbscan = db
            else:
                continue
    return best_dbscan, best_dbscan.labels_


def KMeans_param_search(input_data, cluster_num_min, cluster_num_max):
    score = -1
    best_model = KMeans()
    for i in range(cluster_num_min, cluster_num_max, 1 ):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init='auto', random_state=0)
        kmeans.fit(input_data)
        labels = kmeans.labels_
        if score <= silhouette_score(input_data, labels):
            score = silhouette_score(input_data, labels)
            best_model = kmeans
    print(best_model.n_clusters, score)
    return best_model, best_model.labels_


def AgglomerativeClustering_param_search(input_data, cluster_num_min, cluster_num_max):
    input_data = np.array(input_data, dtype=np.float32)
    score = -1
    best_model = AgglomerativeClustering()
    # at_least = math.ceil(cluster_num // 4 * 3) + 2
    for n_clusters in range(cluster_num_min, cluster_num_max, 1):  # math.ceil(2.5* cluster_num), 3* cluster_num+10
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        agg_clustering.fit(input_data)
        labels = agg_clustering.labels_
        if score <= silhouette_score(input_data, labels):
            score = silhouette_score(input_data, labels)
            best_model = agg_clustering
    print(best_model.n_clusters, score)
    return best_model, best_model.labels_


def cluster_discovery(parameters, tableNames):
    if not parameters:
        print("parameters are invalid! Check the code.")
        return None
    labels = parameters[1]
    l = (tableNames, labels)
    clust = zip(*l)
    clu = list(clust)
    return clu


def cluster_Dict(clusters_list):
    cluster_dictionary = {}
    for k, v in clusters_list:
        if cluster_dictionary.get(v) is None:
            cluster_dictionary[v] = []
        cluster_dictionary[v].append(k)
    return cluster_dictionary


def clustering(input_data, data_names, number_estimate, clustering_method):
    parameters = []
    min_num = number_estimate - 3
    max_num = 2 * number_estimate
    if clustering_method == "DBSCAN":
        parameters = dbscan_param_search(input_data)
    if clustering_method == "Agglomerative":
        parameters = AgglomerativeClustering_param_search(input_data, min_num, max_num)
    if clustering_method == "KMeans":
        parameters = KMeans_param_search(input_data, min_num, max_num)
    clusters = cluster_discovery(parameters, data_names)
    cluster_dict = cluster_Dict(clusters)
    return cluster_dict


def result_precision(clustering_dict: dict, isStrict=True):
    gt = pd.read_csv(os.path.join(current_dir_path, "groundTruth.csv"))
    for cluster_id, tables in clustering_dict.items():
        class_tables = []
        for table in tables:
            if isStrict is True:
                class_table = gt[gt['fileName'] == table].iloc[0, 2]
            else:
                class_table = gt[gt['fileName'] == table].iloc[0, 3]
            class_tables.append(class_table)
        cluster_label, highest_frequency = find_most_frequent_elements(class_tables)
        precision = 1 - len([i for i in class_tables if i not in cluster_label]) / len(class_tables)
        if len(cluster_label) == 1:
            cluster_label = cluster_label[0]
        print(cluster_id, cluster_label, precision)


def typeInference(embedding_file_path, clustering_method="Agglomerative", numEstimate=0):
    dict_file = {}
    F = open(embedding_file_path, 'rb')
    content = pickle.load(F)
    Z = []
    T = []
    # for showing the first item in content
    for vectors in content:
        T.append(vectors[0])
        # use average column embeddings in a table to indicate the whole table embedding
        vec_table = np.mean(vectors[1], axis=0)
        Z.append(vec_table)
    Z = np.array(Z)
    cluster_dict = clustering(Z, T, number_estimate=numEstimate, clustering_method=clustering_method)
    return cluster_dict

def conceptualAttri(dataset_path: str, embedding_file_path: str, clustering_method="KMeans", domain="VideoGame",
                    numEstimate=0):
    Z = []
    T = []
    F = open(embedding_file_path, 'rb')
    # content is the embeddings for all datasets
    content = pickle.load(F)
    gt = pd.read_csv(os.path.join(current_dir_path, "groundTruth.csv"))
    domain_table_names = list(gt[gt['class'] == domain].iloc[:, 1])
    content = [vector for vector in content if vector[0] in domain_table_names]
    for table_name, vectors in content:
        headers = pd.read_csv(os.path.join(dataset_path, table_name)).columns
        vectors = np.array(vectors)
        for index, header in enumerate(headers):
            T.append(f"{table_name}.{header}")
            Z.append(vectors[index])
    cluster_dict = clustering(Z, T, number_estimate=numEstimate, clustering_method=clustering_method)
    cluster_dict = dict(sorted(cluster_dict.items()))
    return cluster_dict


