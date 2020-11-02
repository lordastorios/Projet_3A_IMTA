from classes import*
import numpy as np
from math import *
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def V_difference(V1,V2):
    diff=0
    diff+=np.sum((V1.C-V2.C)**2)
    diff+=np.sum((V1.L-V2.L)**2)
    diff+=(V1.D-V2.D)**2
    diff+=(V1.w-V2.w)**2
    diff+=abs(V1.d-V2.d)*V1.w*V2.w
    return diff

def S_difference(Mod,Obs):
    if (Obs.NumVer==0 or Mod.NumVer==0) and Obs.NumVer!=Mod.NumVer:
        return Inf
    diff=0
    S1,S2=Mod,Obs
    if Mod.NumVer>Obs.NumVer:
        S1,S2=Obs,Mod
    for v1 in S1.Vertices:
        d=[]
        for v2 in S2.Vertices:
            d.append(V_difference(v1,v2))
        diff+=min(d)
    return min(diff,20)

def minimalSituation(Mod,Obs):
    Diff=[]
    for obs in Obs:
        Diff.append(S_difference(Mod,obs))
    Best_S=np.array([[],[]])
    for i in range(1,len(Obs)-1):
        if Diff[i]<Diff[i-1] and Diff[i]<Diff[i+1]:
            Best_S=np.concatenate((Best_S,np.array([[int(i)],[Diff[i]]])),axis=1)
    Best_S[1,:]=(1/Best_S[1,:])/np.sum(1/Best_S[1,:]) # compute ressemblance
    return Best_S

def next_vortex(v,simV,wV,followV):
    C,L,D,d,w,error=v.C,v.L,v.D,v.d,v.w,v.error
    for i in range(len(simV)):
        v_,f_,w_=simV[i],followV[i],wV[i]
        C,L,D,w=C+w_*(f_.C-v_.C),L+w_*(f_.L-v_.L),D+w_*(f_.D-v_.D),w+w_*(f_.w-v_.w)
    return Vortex(C,L,D,d,w,error)

def next_model(Mod,Obs,BS):
    next_V=[]
    for v in Mod.Vertices:
        simV=[]
        wV=[]
        followV=[]
        for i in range(np.shape(BS)[1]):
            s=Obs[int(BS[0,i])]
            l=[V_difference(v,v_) for v_ in s.Vertices]
            if min(l)>500:
                break
            vi=l.index(min(l))
            followS=Obs[int(BS[0,i])+1]
            followL=[V_difference(s.getVortex(vi),v_) for v_ in followS.Vertices]
            if min(followL)>200:
                break
            simV.append(s.getVortex(vi))
            wV.append(BS[1,i])
            followV.append(followS.getVortex(followL.index(min(followL))))
        if len(simV)==0:
            next_V.append(v)
            break
        next_V.append(next_vortex(v,simV,wV,followV))
    return Situation(next_V)