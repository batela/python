'''
Created on 18.06.2009

@author: Lars Vogel
'''
import numpy as np
import sklearn.neighbors as skln
from sklearn.neighbors import KNeighborsRegressor
import CDaytypeForecaster
import CDayTypeClasificator


# def add(a,b):
#     pylab.close()
#      
#     # generate 3 sets of normally distributed points around
#     # different means with different variances
#     pt1 = numpy.random.normal(1, 0.2, (100,2))
#     pt2 = numpy.random.normal(2, 0.5, (300,2))
#     pt3 = numpy.random.normal(3, 0.3, (100,2))
#     
#     print np.shape(pt1)
#     print np.shape(pt2)
#     print pt3 
#     # slightly move sets 2 and 3 (for a prettier output)
#     pt2[:,0] += 1
#     pt3[:,0] -= 0.5
#      
#     xy = numpy.concatenate((pt1, pt2, pt3))
#     print np.shape(xy) 
#     # kmeans for 3 clusters
#     res, idx = kmeans2(numpy.array(zip(xy[:,0],xy[:,1])),3)
#     print idx 
#     colors = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in idx])
#      
#     # plot colored points
#     pylab.scatter(xy[:,0],xy[:,1], c=colors)
#      
#     # mark centroids as (X)
#     pylab.scatter(res[:,0],res[:,1], marker='o', s = 500, linewidths=2, c='none')
#     pylab.scatter(res[:,0],res[:,1], marker='x', s = 500, linewidths=2)
#      
#     pylab.savefig('/tmp/kmeans.png')
# True


if __name__ == '__main__':
    
    print "INICIO"
    
#     add(1,2)

	# Nueva linea que meto en el codigo
   
    dtClasificator = CDayTypeClasificator.CDayTypeClasificator("")
    dtClasificator.createCluster()
    dtClasificator.readInputValues()
   
    rawData = np.genfromtxt('/home/borja/Prj/Beest/Tests/testtrain0.csv',delimiter=',')
    nbrs = skln.NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(rawData)
    X = np.array([0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
    distances, indices = nbrs.kneighbors(X)
    
    print distances
    print indices
   
    dtForecaster = CDaytypeForecaster.CDaytypeForcaster("")
    prdic = dtForecaster.predictDayType(3, 4)

    


#rawData = np.genfromtxt('/home/borja/Prj/Beest/Tests/CalendarTest.txt',delimiter=';',usecols=(0,1))
#print rawData

#solData = np.genfromtxt('/home/borja/Prj/Beest/Tests/CalendarTest.txt',delimiter=';',usecols=(2))
#print solData
#norm_data=np.array([my_data[0,:],my_data[1,:],my_data[2,:]])

# normData=np.array([ rawData[:,0]/np.linalg.norm(rawData[:,0]),rawData[:,1]/np.linalg.norm(rawData[:,1]),rawData[:,2]/np.linalg.norm(rawData[:,2]) ])
# print "Normalizado"
# print normData.shape

# normMt=np.matrix([ rawData[:,0]/np.linalg.norm(rawData[:,0]),rawData[:,1]/np.linalg.norm(rawData[:,1]),rawData[:,2]/np.linalg.norm(rawData[:,2]) ])

# print "Matrix"
# print normMt

# print "Traspuesta"
# print normMt.T


# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2],[-1, -1.1]])
# print X.shape
#nbrs = skln.NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(rawData,solData)

#knn = KNeighborsRegressor(n_neighbors=5)
#knn.fit(rawData, solData)
# Make point predictions on the test set using the fit model.


# distances, indices = nbrs.kneighbors(X)
# print "Indices"
# print indices
# print "Distancias"
# print distances


#X = np.array([2,5])   
  
#predictions = knn.predict(X)
#print predictions

#distances, indices = nbrs.kneighbors(X)
#print "Indices"
#print indices
#print "Distancias"
#print distances

