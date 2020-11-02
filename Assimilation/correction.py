from classes import*
import numpy as np
from math import *
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from prediction import V_difference

def isMatching(V_,V):
    if np.linalg.norm(V_.C-V.C)>V.error[0]+V_.error[0]: return False
    if (abs(V_.L[0]-V.L[0])>V_.L[0]*V_.error[1]+V.L[0]*V.error[1]) or (abs(V_.L[1]-V.L[1])>V_.L[1]*V_.error[1]+V.L[1]*V.error[1]): return False
    if abs(V_.D-V.D)>360*V_.error[2]+360*V.error[2]: return False
    if (V_.d+V.d==0) and (V_.error[3]+V.error[3]!=2):return False
    if abs(V_.w-V.w)>V_.w*V_.error[4]+V.w*V.error[4]:return False
    return True

def correctVortex(V_,V):
    W_=(np.ones((1,5))/V_.error)/((np.ones((1,5))/V_.error)+(np.ones((1,5))/V.error))
    W=np.ones((1,5))-W_
    C=W_[:,0]*V_.C+W[:,0]*V.C
    L=W_[:,1]*V_.L+W[:,1]*V.L
    D=W_[:,2]*V_.D+W[:,2]*V.D
    d=V.d
    w=W_[:,4]*V_.w+W[:,4]*V.w
    return Vortex(C,L,D,d,w,np.array([0.3,0.3,0.3,0.3,0.3]))

def correctModel(Mod,Obs):
    V_,V=Mod.Vertices.copy(),Obs.Vertices.copy()
    NewV=[]
    for v_ in V_:
        match=False
        for v in V:
            if V_difference(v_,v)<500:
            #if isMatching(v_,v):
                NewV.append(correctVortex(v_,v))
                V.remove(v)
                match=True
                break
        if match==False:
            NewV.append(v_)
    NewV=NewV+V
    return Situation(NewV)