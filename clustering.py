import numpy as np	
import pyclustering as pc
import matplotlib.pyplot as plt

from numpy import genfromtxt
from pyclustering.cluster.cure import cure
from pyclustering.cluster import cluster_visualizer_multidim

#Read data
def read_data(path, read_as = 'list'):
	path = 'Data/Dataset(Clustering).csv'
	if read_as == 'pandas_dataframe' :
		data = pd.read_csv(path, delimiter=";")
	else:
		data = genfromtxt('Data/Dataset(Clustering).csv', delimiter=';')
		data = np.delete(data, 0, 0)

	if read_as == 'list':
		data = data.tolist()

	return data

#Visualize clusters in multi-dimensional space
def visualize_cluster(clusters, input_data, pair_filter=[[0, 1], [0, 2]]):
	visualizer = cluster_visualizer_multidim() 
	visualizer.append_clusters(clusters, input_data);
	visualizer.show(pair_filter);

if __name__ == '__main__':
	
	input_data = read_data('Data/Dataset(Clustering).csv')
	# CURE Algorithm 
	cure_instance = cure(input_data, number_cluster = 4, number_represent_points = 5, compression = 0.5, ccore = True);	
	cure_instance.process();
	#Returns list of allocated clusters, each cluster contains indexes of objects in list of data.
	clusters = cure_instance.get_clusters();
	#Returns list of point-representors of each cluster
	representatives_points = cure_instance.get_representors()
	#Mean of points that make up each cluster (Centroid of each cluster)
	means = cure_instance.get_means() 



	

	
	


	
	
	

	
