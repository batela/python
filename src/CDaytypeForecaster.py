# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:50:05 2015

@author: 105083
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class CDaytypeForcaster():
    
    rawData=[]
    dayType=[]
    
    
    def __init__ (self,rp):
        self.repopath = rp
        
    def readInputValues (self):
        rawData = np.genfromtxt('/home/borja/Prj/Beest/Tests/CalendarTest.txt',delimiter=';',usecols=(0,1))
        dayType = np.genfromtxt('/home/borja/Prj/Beest/Tests/CalendarTest.txt',delimiter=';',usecols=(2))
     
    def predictDayType (self,week,day):
        
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(self.rawData, self.dayType)

        X = np.array([week,day])   
        predictions = knn.predict(X)
        return predictions
