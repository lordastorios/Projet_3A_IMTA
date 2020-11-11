import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Catalog:
    #define a set of situations
    def __init__(self,Situations):
        self.Situations=Situations
        self.NumSit=len(Situations)

    # return the list of distance to the situation S for each situation of the catalog
    def C_distance(self,S):
        dist=[]
        for s in self.Situations:
            dist.append(S.S_distance(s))
        return dist

    # return two dictionnary:
    #       - local closest situation (key) and their successor (value)
    #       - local closest situation (key) and their weight (value)
    # Note: closest neighbors are here local minimum of the dist values
    # Note: computation of weights in function of distance can be improve...
    def Neighbors(self,S,disp):
        successor={}
        weight={}
        dist=self.C_distance(S)
        if disp==True:
            f=plt.figure(0)
            plt.plot(range(self.NumSit),dist)
        w_=0
        l=[]
        for i in range(1,self.NumSit-1):
            if dist[i]<dist[i-1] and dist[i]<dist[i+1]:
                w_+=1/(1+dist[i])
                l.append(i)
        for i in l:
            successor[self.Situations[i]]=self.Situations[i+1]
            weight[self.Situations[i]]=1/(1+dist[i])*1/w_
        return successor, weight


class Situation:
    # define a set of vertices
    def __init__(self,Vertices):
        self.Vertices=Vertices
        self.NumVer=len(Vertices)

    # compute the distance between with an other situation S:
    #       - for each vortex, search the more similar vortex in the other situation
    #       - add the distance between these two vertices to the total distance
    #       - upper bound (here 20) to choose, we do not consider situation further than that
    # Note: to improve...
    def S_distance(self,S):
        dist=0
        if (self.NumVer==0 or S.NumVer==0) and self.NumVer!=S.NumVer:
            return math.inf
        for u in self.Vertices:
            d=[]
            for v in S.Vertices:
                d.append(u.V_distance(v))
            dist+=min(d)
        return min(dist,20)

    # return a corrected situation of the model considering an observed situation:
    #       - search the most similar vortex and do the correction in consequence
    #       - a vortex whth no similar vortex (max 500 here) is unchanged
    def S_correction(self,S):
        if self.NumVer>S.NumVer:
            U,V=self.Vertices,S.Vertices
        else:
            V,U=self.Vertices,S.Vertices
        new_V=[]
        for u in U:
            min_dist=math.inf
            for v in V:
                dist=u.V_distance(v)
                if dist<500 and dist<min_dist:
                    min_dist=dist
                    u_=v
            if min_dist<500:
                new_V.append(u.V_correction(u_))
            else:
                new_V.append(u)
        return Situation(new_V)

    # compute the next situation using prediction on each vortex:
    #       - 500 is here a const above ich we consider vertices are not similar
    def S_prediction(self,successor,weight):
        next_V=[]
        for v in self.Vertices:
            simV=[]
            wV=[]
            followV=[]
            for s in successor:
                l=[v.V_distance(u) for u in s.Vertices]
                if min(l)>500: # break if no similar vortex
                    break
                vi=l.index(min(l))
                followS=successor[s]
                followL=[s.getVortex(vi).V_distance(u) for u in followS.Vertices]
                if min(followL)>500:# break if the vortex disappear between t and t+1
                    break
                simV.append(s.getVortex(vi))
                wV.append(weight[s])
                followV.append(followS.getVortex(followL.index(min(followL))))
            if len(simV)==0:
                next_V.append(v)
                break
            next_V.append(v.V_prediction(simV,wV,followV))
        return Situation(next_V)

    # return vortex of index i
    def getVortex(self,i):
        return self.Vertices[i]

    # add a vortex to the situation
    def addVortex(self,Vortex):
        self.Vertices.append(Vortex)
        self.NumVer+=1

    # remove a vortex from the situation
    def removeVortex(self,Vortex):
        self.Vertices.remove(Vortex)
        self.NumVer-=1

    # set up the plot of each vortex
    def S_map(self):
        m=[]
        for v in self.Vertices:
            m.append(v.V_map())
        return m

class Vortex:
    # define a vortex
    def __init__(self,C,L,D,d,w,error):
        self.C=C # cluster center (np.array(2))
        self.L=L # cluster ellipse axis lenghts (np.array(1,2))
        self.D=D # cluster ellipse axis direction (real)
        self.d=d # vortex rotation direction (wise)
        self.w=w # vortex angular velocity (real)
        self.error=error # list error on each variable ([5])

    # compute the distance to an other vortex
    # Note: really basic for the moment, we have to work on it...
    def V_distance(self,V):
        dist=0
        dist+=np.sum((self.C-V.C)**2)
        dist+=np.sum((self.L-V.L)**2)
        dist+=(self.D-V.D)**2
        dist+=(self.w-V.w)**2
        dist+=abs(self.d-V.d)*V.w*V.w
        return dist

    # return a corrected vortex taking into account an observed vortex V:
    #       - compute weights (W and W_) in fonction of error on each parameter for both vortex
    #       - merge vertices to create a new one
    # Note: for now, the error of the new vortex is still to define
    def V_correction(self,V):
        W_=(np.ones((1,5))/self.error)/((np.ones((1,5))/self.error)+(np.ones((1,5))/V.error))
        W=np.ones((1,5))-W_
        C=W_[:,0]*self.C+W[:,0]*V.C
        L=W_[:,1]*self.L+W[:,1]*V.L
        D=W_[:,2]*self.D+W[:,2]*V.D
        d=V.d
        w=W_[:,4]*self.w+W[:,4]*V.w
        return Vortex(C,L,D,d,w,np.array([0.3,0.3,0.3,0.3,0.3]))

    # return the new vortex using similar vortex (V list) their weights (W list):
    #       - each parameters independly (weighted)
    #       - linear evolution during voretx at time t (V list) and t+1 (F list)
    # Note: we have to find a way of compute a new error
    def V_prediction(self,V,W,F):
        C,L,D,d,w,error=self.C,self.L,self.D,self.d,self.w,self.error
        for i in range(len(V)):
            v,f,w_=V[i],F[i],W[i]
            C,L,D,w=C+w_*(f.C-v.C),L+w_*(f.L-v.L),D+w_*(f.D-v.D),w+w_*(f.w-v.w)
        return Vortex(C,L,D,d,w,error)

    # set up the plot of the vortex:
    #       - center, size and direction
    #       - opacity (alpha) stand for the angular velocity
    #       - color (blue or red) represent the wise
    def V_map(self):
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