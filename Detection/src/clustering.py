from eddies_detection import  get_traj_with_parcels, get_traj_with_scipy, find_eddies, find_eddies2, optimized_streamline
from plot_tools import StreamPlot
import numpy as np
from math import *

def create_list_data(list_eddies):
    """With a list of eddies, returns a list of the eddies' centers and the number of eddies at the first day  """
    K = len(list_eddies[0])
    list_center = []
    for eddies in list_eddies[0]:
        list_center.append(eddies.center)
    return K,list_center


def calc_tho(l):
    """Compute the barycenter for a list l of point"""
    S = np.array([0. for n in range(len(l[0]))])
    for x in l:
        S += np.array(x)
    return S/len(l)

def quadra_d (X,Y):
    """quadratic distance"""
    S = 0
    for i in range(len(X)):
        S += (X[i]-Y[i])**2
    return sqrt(S)

def attrib_cluster(X,tho,d):
    """attribute a cluster to X according to the barycentyer of the clusters tho and the distance d"""
    distances = []
    for t in tho:
        dis = d(X,t)
        distances.append(d(X,t))
    i_min_d = distances.index(min(distances))
    return i_min_d


def Kmean(distance,list_eddies):
    """Implementation of the Kmean algorithm.
    Returns the list of the classified eddies and the day corresponding, 
    the unclassified eddies (not present at day 0) and their date)
    """
    K,list_center = create_list_data(list_eddies)
    
    cluster = [ [] for i in range(K)]
    tho = list_center[:K]
    day_eddies = [ [] for i in range(K)] # list which will save the date of each eddies
    
    tho = tho
    tho_prime = [ 0 for i in range(K)]
    tour = 0    
    
    while ( np.any(np.array(tho_prime) != np.array(tho)) and tour<1000):
        tour += 1
        
        tho_prime = tho.copy()
        cluster = [ [] for i in range(K)]
        day_eddies = [ [] for i in range(K)]
        
        for i_day in range(len(list_eddies)):
            day = list_eddies[i_day]
            for eddies in day:
                X = eddies.center
                i_min_d = attrib_cluster(X,tho,distance)
                cluster[i_min_d].append(X)
                day_eddies[i_min_d].append(i_day)
                
        for  i in range(K):
            tho[i] = calc_tho(cluster[i])
        
        cluster, day_eddies, unclass_eddies, day_unclass_eddies = clean_cluster(list_eddies, cluster, day_eddies,tho, distance)
        
        for i_eddy in range(len(unclass_eddies)):
            tho.append(unclass_eddies[i_eddy])
            cluster.append([unclass_eddies[i_eddy]])
            day_eddies.append([day_unclass_eddies[i_eddy]])
            K+=1
    
        
    return tho, cluster, day_eddies

def clean_cluster(list_eddies, clusters, day_eddies, tho, distance):
    """This function clean the cluster"""
    nb_day = len(list_eddies) # we can't have more eddies in cluster than the number of day we are running the simulation.
    new_clusters = []
    new_day_eddies = []
    unclass_eddies = []
    day_unclass_eddies = []
    for i_cluster in range(len(clusters)):
        n = len(clusters[i_cluster])
        cluster = clusters[i_cluster]
        days = day_eddies[i_cluster]
        if n>nb_day:
            for i in range(n-nb_day):
                i_max = 0
                d_max = 0
                for i_eddy in range(len(cluster)):
                    eddy = cluster[i_eddy]
                    dis = distance(eddy,tho[i_cluster])
                    if dis> d_max:
                        d_max = dis
                        i_max = i_eddy
                        
                foreign_eddy = cluster.pop(i_max)
                day_foreign_eddy = days.pop(i_max)
                unclass_eddies.append(foreign_eddy)
                day_unclass_eddies.append(day_foreign_eddy)
                
        new_clusters.append(cluster)
        new_day_eddies.append(days)
    return new_clusters, new_day_eddies, unclass_eddies, day_unclass_eddies

def predict_d1(cluster):
    prediction = []
    for j in range(len(cluster)):
        n = cluster[j][:-1]
        new_p = [-1,-1]
        if len(n)>2:
            for i in range(len(n)-1):
                vect.append(n[i+1]-n[i])
            vect_m = sum(vect)/len(n)
            newp = n[-1]+vect_m
        prediction.append(newp)
    return prediction


