import numpy as np
from math import *
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Situation:
    def __init__(self,Vertices):
        self.Vertices=Vertices
        self.NumVer=len(Vertices)

    def getVortex(self,i):
        return self.Vertices[i]

    def addVortex(self,Vortex):
        self.Vertices.append(Vortex)
        self.NumVer+=1

    def removeVortex(self,Vortex):
        self.Vertices.remove(Vortex)
        self.NumVer-=1

    def map(self):
        m=[]
        for v in self.Vertices:
            m.append(v.map())
        return m

class Vortex:
    def __init__(self,C,L,D,d,w,error):
        #self.Si=Si # streamline centers (np.array((n,2)))
        self.C=C # cluster center (np.array(2))
        #self.Mk=Mk # cluster covariance (np.array((2,2)))
        self.L=L # cluster ellipse axis lenghts (np.array(1,2))
        self.D=D # cluster ellipse axis direction (np.array(2,1))
        self.d=d # vortex rotation direction (wise)
        self.w=w # vortex angular velocity (real)
        self.error=error # list error on each variable ([(5)])

    def map(self):
        e=Ellipse(xy=self.C, width=self.L[1], height=self.L[0], angle=self.D)
        e.set_alpha(float(self.w/100))
        if self.d==Wise.Clock:
            e.set_facecolor([0,0,1])
        else:
            e.set_facecolor([1,0,0])
        return e

class Wise:
    Clock=-1
    CounterClock=1