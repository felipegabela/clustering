import numpy as np	
import matplotlib.pyplot as plt
from pyclustering.cluster.cure import cure
from pyclustering.cluster import cluster_visualizer_multidim
from scipy.spatial import distance
from sklearn.decomposition import PCA

def read_data(path):
	data = np.genfromtxt(path, delimiter=';')
	data = np.delete(data, 0, 0)
	data = data.tolist()
	return data

# Calculates the average distance to centroid of all clusters
def avg_dist_to_centroid(clusters, centroids, input_data):
	avg_dist = []
	_cluster = 0 
	for cluster in clusters:
		sum_of_distances = 0
		points_cluster = len(cluster)
		for index in range(0, points_cluster):
			point = input_data[cluster[index]]
			centroid =  centroids[_cluster]
			sum_of_distances += distance.euclidean(point, 
				centroid)
		avg_dist.append(sum_of_distances/points_cluster)
		_cluster += 1
	return sum(avg_dist)/len(avg_dist)

# Visualize obtained clusters in multi-dimensional space
def visualize_multidim():
	visualizer = cluster_visualizer_multidim()
	visualizer.append_clusters(clusters, sample_4d)
	visualizer.show(max_row_size=3)


if __name__ == '__main__':

	input_data = read_data('Data/Dataset(Clustering).csv')
	#Delete Outliers
	#outliers = [524, 512, 435, 432, 424, 387, 373, 285, 271, 213, 167, 147, 60, 47, 44, 40] #522, 213
	#outliers =[524, 40, 522, 213] #Aaron
	outliers = [524, 512, 435, 432, 424, 387, 373, 285, 271, 213, 167, 147, 60, 47, 44 ] #Adrian

	input_data = np.delete(input_data, outliers, 0)

	k_d_plot = {}
	k_centroids = {}
	k_clusters = {}
	for k in range(1,11):
		# Define parameters for the CURE algorithm 
		cure_instance = cure(
			input_data, number_cluster=k, 
			number_represent_points=4, 
			compression=0.2, ccore=True);
		# Run the algorithm	
		cure_instance.process();
		# Get list of allocated clusters.
		# Each cluster contains indexes of objects in list of data.
		clusters = cure_instance.get_clusters();
		# Returns list of point-representors of each cluster
		representatives_points = cure_instance.get_representors()
		# Mean of points that make up each cluster
		centroids = cure_instance.get_means()
		#Clusters average distance to centroids
		avg_dist = avg_dist_to_centroid(clusters, 
			centroids, input_data)
		
		#K-Clusters vs Avg Distance to Cluster
		k_d_plot[k] = avg_dist
		#Appending centroid value of each cluster for each different value of K
		k_centroids[k] = centroids
		#List of clusters with the indexes of objects for each K 
		k_clusters [k] = clusters

	print('K-Clusters vs Avg. Distance to Centroid:\n', k_d_plot)
	tmp = sorted(k_d_plot.items()) 
	k, dist = zip(*tmp) 
	plt.plot(k, dist)
	plt.xlabel('K-Clusters')
	plt.ylabel('Avg Distance to Centroid')
	plt.show()

	#Solucion Optima
	optimal_k = 2
	print('\nOptimal Solution K = ', optimal_k)
		
	#Centroid of each cluster
	for i in range (0, len(k_centroids[optimal_k])):
		print('\nCentroid Cluster ', 
			i+1, ':\n', k_centroids[optimal_k][i])
		
	#Number of observations in each cluster
	for i in range (0, len(k_clusters[optimal_k])):
		print('Number of instances in Cluster ', 
				i+1, ': ', len(k_clusters[optimal_k][i]))		
