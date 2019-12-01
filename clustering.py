import numpy as np	
import matplotlib.pyplot as plt
from pyclustering.cluster.cure import cure
from pyclustering.cluster import cluster_visualizer_multidim
from scipy.spatial import distance

def read_data(path, read_as='list'):
	path = 'Data/Dataset(Clustering).csv'
	if read_as == 'pandas_dataframe':
		data = pd.read_csv(path, delimiter=";")
	else:
		data = np.genfromtxt('Data/Dataset(Clustering).csv', 
			delimiter=';')
		data = np.delete(data, 0, 0)

	if read_as == 'list':
		data = data.tolist()

	return data

# Visualize clusters in multi-dimensional space
def visualize_cluster(
		clusters, input_data, 
		pair_filter=[[0, 1], [0, 2]]
		):
	visualizer = cluster_visualizer_multidim() 
	visualizer.append_clusters(clusters, input_data);
	visualizer.show(pair_filter);

# Calculates the average distance to centroid of each cluster
def avg_dist_to_centroid(clusters, centroids, input_data):
	avg_dist = []
	_cluster = 0 
	for cluster in clusters:
		sum_of_distances = 0
		points_cluster = len(cluster)
		for index in range(0, points_cluster):
			point = cluster[index]
			centroid =  centroids[_cluster]
			sum_of_distances += distance.euclidean(point, 
				centroid)
		avg_dist.append(sum_of_distances/points_cluster)
		_cluster += 1
	return avg_dist


if __name__ == '__main__':
	
	k_d_plot = {}
	for k in range(1,11):
		input_data = read_data('Data/Dataset(Clustering).csv')
		# Define parameters for the CURE algorithm 
		cure_instance = cure(
			input_data, number_cluster=k, 
			number_represent_points=4, 
			compression=0.2, ccore=True);
		# Run the algorithm	
		cure_instance.process();
		'''
		# Get list of allocated clusters.
		# Each cluster contains indexes of objects in list of data.
		'''
		clusters = cure_instance.get_clusters();
		# Returns list of point-representors of each cluster
		representatives_points = cure_instance.get_representors()
		# Mean of points that make up each cluster
		centroids = cure_instance.get_means()
		#Clusters average distance to centroids
		avg_dist = avg_dist_to_centroid(clusters, 
			centroids, input_data)
		k_d_plot[k] = avg_dist
	print('K-Custers vs Avg. Distance to Centroid:\n\n', k_d_plot)




	




	

	
	


	
	
	

	
