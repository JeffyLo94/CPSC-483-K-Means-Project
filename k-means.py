from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt

# K Means Functions

def calc_euclidean_dist( point, centroid ):
    # print(point)
    # print(centroid)
    # print('dist: ', math.sqrt(((point[0] - centroid[0])**2) + ((point[1] - centroid[1])**2)))
    return math.sqrt(((point[0] - centroid[0])**2) + ((point[1] - centroid[1])**2))

def initial_plot( data, centroids, colors=['red','blue','green'] ):
    plt.figure(1)
    # plt.scatter(data)
    x= data[:,0]
    y= data[:,1]
    # print('x: ', x)
    # print('y: ', y)

    # plot points
    plt.scatter(x, y, s=10, c='black', marker='.')

    colorCount = 0
    #plot centroids
    for pt in centroids:
        plt.scatter(pt[0], pt[1], s=30, edgecolors=colors[colorCount], marker='^', facecolors='none')
        if(colorCount == len(colors)):
            colorCount=0
        else:
            colorCount += 1

    plt.suptitle('Initial Plot')
    plt.axis([0,9,0,9])


def generate_plot_from_plot_data(plot_data, centroids, title='', pltNum=1, colors=['red','blue','green']):
    plt.figure(pltNum)
    clustered_points = get_points_from_cluster_data(plot_data)
    # print(clustered_points)
    colorCount = 0
    for data_points in clustered_points:
        x = data_points[:,0]
        y = data_points[:,1]
        plt.scatter(x, y, s=10, c=colors[colorCount], marker='.')
        plt.scatter(centroids[colorCount][0], centroids[colorCount][1], s=30, edgecolors=colors[colorCount], marker='^', facecolors='none')
        colorCount += 1

    plt.suptitle(title)
    plt.axis([0,9,0,9])

def generate_next_centroids( cluster_data, k=3 ):
    new_centroids =[ None for _ in range(0,k)]
    for cluster in cluster_data:
        points = []
        # gather points from cluster
        for pt_info in cluster_data[cluster]:
            points.append(pt_info['POINT'])
            i = pt_info['C INDEX']
        # print(type(points[0][1]))
        np_points = np.array(points, dtype=float)
        # print(np_points)
        new_centroids[i] = np_points.mean(axis=0)
    # print(np.array(new_centroids))
    # print(type(new_centroids[0][0]))
    return np.array(new_centroids)

def get_points_from_cluster_data( cluster_data, k=3 ):
    clustered_points = [None for _ in range(0,k)]
    for cluster in cluster_data:
        points = []
        # gather points from cluster
        for pt_info in cluster_data[cluster]:
            points.append(pt_info['POINT'])
            i = pt_info['C INDEX']
        # print(type(points[0][1]))
        np_points = np.array(points, dtype=float)
        clustered_points[i] = np_points
    return clustered_points

def assign_point_to_cluster( point, distance_map, centroids):
    pt_cluster_info = {}
    # print('min: ', min(distance_map, key=distance_map.get))
    min_cluster_key = min(distance_map, key=distance_map.get)
    # print('min key: ', min_cluster_key)
    cluster_points = []
    # numpy float strings have spaces sometimes
    for x in min_cluster_key.strip('][').split(', '):
        if x != '':
            cluster_points.append(float(x))
    min_cluster = cluster_points
    # print('min cluster: ', min_cluster)
    pt_cluster_info['KEY'] = min_cluster_key
    pt_cluster_info['CLUSTER'] = min_cluster
    centroid_index = -1
    for i in range(len(centroids)):
        # print(centroids[i].tolist())
        # print(min_cluster)
        if centroids[i].tolist() == min_cluster:
            centroid_index = i
    pt_cluster_info['C INDEX'] = centroid_index
    pt_cluster_info['POINT'] = point
    # print('cluster result: ', pt_cluster_info)
    return pt_cluster_info


def k_means( data, centroids, max_iterations ):
    cs = centroids
    first_iteration_results = {}
    last_iteration_results = {}
    first_iteration_centroids = None
    last_iteration_centroids = None

    iteration_point_clusters = {}

    num_data_points = len(data)
    k = len(centroids)
    print('k =', k)

    for i in range(0, max_iterations):
        # reset iteration cluster map
        for c in cs:
            iteration_point_clusters[str(c.tolist())] = []

        for pt in data:
            dist_map = {}
            for c in cs:
                # print('c:', c)
                dist_map[str(c.tolist())] = calc_euclidean_dist(pt, c)
                # print('distances: ', dist_map)
            pt_cluster_info = assign_point_to_cluster(pt, dist_map, cs)

            iteration_point_clusters[pt_cluster_info['KEY']].append(pt_cluster_info)
            # Stash data for plot
            # if i == 0:
            #     first_iteration_results.append(pt_cluster_info)
            #     # print(first_iteration_results)
            # elif i == max_iterations-1:
            #     last_iteration_results.append(pt_cluster_info)
            #     # print(last_iteration_results)

        # Update centroids
        cs = generate_next_centroids(iteration_point_clusters)
        print('updated centroids for iter #' + str(i) + ':')
        print(cs)

        # stash data for plot
        if i==0:
            first_iteration_results = copy.deepcopy(iteration_point_clusters)
            first_iteration_centroids = cs
        elif i == max_iterations-1:
            last_iteration_results = copy.deepcopy(iteration_point_clusters)
            last_iteration_centroids = cs

    return {
        'FIRST_ITERATION': first_iteration_results,
        'FIRST_CENTROIDS': first_iteration_centroids,
        'LAST_ITERATION': last_iteration_results,
        'LAST_CENTROIDS': last_iteration_centroids
    }

# M A I N
RUN_ITERATIONS = 10
INITIAL_CENTROIDS = np.array([
    [3, 3],
    [6, 2],
    [8, 5]
])

mat = loadmat("kmeansdata.mat")
# print('printing Initial Data Set:')
# print(mat['X'])

# df = pd.DataFrame(mat['X'])
# print(df)
arr= np.array(mat['X'])
# print(arr)

#INITIAL PLOT
initial_plot(data= arr, centroids=INITIAL_CENTROIDS)

plot_data = k_means( data=arr, centroids=INITIAL_CENTROIDS, max_iterations=RUN_ITERATIONS )

# print('first iteration: ', plot_data['FIRST_ITERATION'])
# print('1st centroids: ', plot_data['FIRST_CENTROIDS'])
generate_plot_from_plot_data(plot_data['FIRST_ITERATION'], plot_data['FIRST_CENTROIDS'], title='FIRST Iteration', pltNum=2)
# print('last iteration: ', plot_data['LAST_ITERATION'])
# print('last centroids: ', plot_data['LAST_CENTROIDS'])
generate_plot_from_plot_data(plot_data['LAST_ITERATION'], plot_data['LAST_CENTROIDS'], title='LAST Iteration', pltNum=3)

# show plot
plt.show()
plt.close()