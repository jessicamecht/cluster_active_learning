from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics.pairwise import euclidean_distances
from itertools import groupby
import operator
import numpy as np

from visualizations.plot_helpers import plot_clusters_3D

def get_boundary_points(data):
    '''
    returns boundary points and their distance to the respective mean to the mean point assigned by KNN
    '''
    #distances_to_mean = KMeans(n_clusters=10).fit_transform(data)
    meanShift = MeanShift(bin_seeding=True).fit(data)
    cluster_centers = meanShift.cluster_centers_
    distances_to_mean = euclidean_distances(data, cluster_centers)
    dist_to_assigned_mean = list(map(lambda i, x: (i, x.tolist().index(min(x)), min(x)), range(len(distances_to_mean)), distances_to_mean))#returns (index, cluster_affiliation, distance_from_mean)
    #for each cluster, get the the indices of 2 instances which are most far away from the cluster center
    grouped_clusters = groupby(sorted(list(dist_to_assigned_mean), key=lambda x: x[1]), operator.itemgetter(1))#group list by cluster affiliation
    idxs = []
    dists = []

    for key, group_object in grouped_clusters:
        sorted_group_list = sorted(list(group_object), key=lambda x: x[-1])
        max_2_dist = sorted_group_list[len(sorted_group_list)-2:len(sorted_group_list)]

        for elem in max_2_dist:
            idx = elem[0]
            dist = elem[-1]
            idxs.append(idx)
            dists.append(dist)
    return idxs, dists

def get_cluster_assignments(data):
    meanshift = MeanShift(bin_seeding=True).fit(data)
    labels = meanshift.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    P = meanshift.predict(data)
    return P

def get_distance_between_known_instances(z, labels, idxs):
    most_confusing_data = [list(z[i]) for i in idxs]
    labels = [labels[i] for i in idxs]
    merged_list = [(most_confusing_data[i], labels[i]) for i in range(len(labels))]

    grouped_clusters = groupby(sorted(merged_list, key=lambda x: x[1]),
                               operator.itemgetter(1))  # group list by label affiliation
    summed_distances = 0
    for key, group_object in grouped_clusters:
        x = list(group_object)
        x = list(map(lambda x: x[0], x))
        distances = euclidean_distances(x, x)
        summed_distances += np.sum(distances)
    return  summed_distances

