'''
Created on May 1, 2016

@author: root
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy
import matplotlib
matplotlib.use('Agg')
from scipy.cluster.vq import kmeans2,kmeans,vq
import pylab

class CDayTypeClasificator(object):
    '''
    classdocs
    '''
    data = []
    nn = None
    
    def __init__(self, params):
        '''
        Constructor
        '''
    def createCluster (self):
        rawData = np.genfromtxt('/home/borja/Prj/Beest/Tests/testtrain.csv',delimiter='\t',usecols=(2))
        data = np.reshape(rawData, (np.shape(rawData)[0]/24,24))
        
        rawType = np.genfromtxt('/home/borja/Prj/Beest/Tests/testtrain.csv',delimiter='\t',usecols=(12))
        dayType = np.reshape(rawType, (np.shape(rawType)[0]/24,24))
        print data
        print dayType
        dayType = dayType[:,0]
        print dayType
        pylab.close()
     
         
        print np.array(zip(rawType,rawData)) 
        #res, idx = kmeans2(np.array(zip(rawType,rawData)),3)
        centroids,_ = kmeans(data,3)
        idx,_ = vq(data,centroids)
        
        
        #res, idx = kmeans2(rawData,3)
        print idx
        print centroids
        colors = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in idx])
         
         
         
#         pylab.plot(rawType,rawData,c=colors)
        pylab.scatter(rawType,rawData,c=colors)
        pylab.savefig('/tmp/kmeans.png')
        # generate 3 sets of normally distributed points around
        # different means with different variances
        pt1 = numpy.random.normal(1, 0.2, (100,2))
        pt2 = numpy.random.normal(2, 0.5, (300,2))
        pt3 = numpy.random.normal(3, 0.3, (100,2))
        
        print np.shape(pt1)
        print np.shape(pt2)
        print pt3 
        # slightly move sets 2 and 3 (for a prettier output)
        pt2[:,0] += 1
        pt3[:,0] -= 0.5
         
        xy = numpy.concatenate((pt1, pt2, pt3))
        print np.shape(xy) 
        # kmeans for 3 clusters
        res, idx = kmeans2(numpy.array(zip(xy[:,0],xy[:,1])),3)
        print idx 
        colors = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in idx])
         
        # plot colored points
        pylab.scatter(xy[:,0],xy[:,1], c=colors)
         
        # mark centroids as (X)
        pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
        pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)
         
        pylab.savefig('/tmp/kmeans.png')

        
        
        
    def readInputValues (self):
        
        rawData = np.genfromtxt('/home/borja/Prj/Beest/Tests/testtrain.csv',delimiter='\t',usecols=(2))
        data = np.reshape(rawData, (np.shape(rawData)[0]/24,24))
        
        dayType = np.genfromtxt('/home/borja/Prj/Beest/Tests/testtrain.csv',delimiter='\t',usecols=(11))
        dayType = np.reshape(dayType, (np.shape(dayType)[0]/24,24))
        print data
        print dayType
        dayType = dayType[:,0]
        print dayType
        
        nn= NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)