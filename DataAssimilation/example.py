import data_assimilation
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

### create a pseudo-random catalog for testing

V1=Vortex(np.array([2,6.5]),np.array([4,2.3]),33,Wise.CounterClock,32,np.array([0.3,0.3,0.3,0.3,0.3]))
V2=Vortex(np.array([4.5,3]),np.array([2,1.4]),102,Wise.Clock,67,np.array([0.3,0.3,0.3,0.3,0.3]))
V3=Vortex(np.array([7.6,8]),np.array([3,3]),31,Wise.CounterClock,72,np.array([0.3,0.3,0.3,0.3,0.3]))

C=[Situation([V3,V1,V2])]

def next_situation(last_situation,bool):
    new_situation=Situation([])
    for j in range(last_situation.NumVer):
        last_vortex=last_situation.getVortex(j)
        if rnd.gauss(0,1)<2.4 or np.amin(last_vortex.L)>1 or last_vortex.w>20:
            C=last_vortex.C+np.random.randn(2)/5
            L=last_vortex.L+np.random.randn(2)/5
            D=last_vortex.D+rnd.gauss(0,0.5)
            w=last_vortex.w+rnd.gauss(0,0.01)
            d,error=last_vortex.d,last_vortex.error
            if bool==True:
                error=[0.1,0.1,0.1,0.1,0.1]
            new_vortex=Vortex(C,L,D,d,w,error)
            new_situation.addVortex(new_vortex)
        if rnd.gauss(0,1)>2.6:
            C=np.array([rnd.random()*10,rnd.random()*10])
            L=np.array([rnd.random(),rnd.random()])
            D=rnd.random()*360
            d=rnd.choice([Wise.Clock,Wise.CounterClock])
            w=abs(rnd.gauss(4,1))
            error=last_vortex.error
            if bool==True:
                error=[0.1,0.1,0.1,0.1,0.1]
            new_vortex=Vortex(C,L,D,d,w,error)
            new_situation.addVortex(new_vortex)
    return new_situation

for i in range(100):
    last_situation=C[-1]
    new_situation=next_situation(last_situation,False)
    C.append(new_situation)

Cata=Catalog(C)

### compute the initiale situation

Model=C[20]
for i in range(4):
    Model=next_situation(Model,bool=False)

### compute some observation

t1,t2,t3,t4=3,8,17,29
Obs1,Obs2,Obs3,Obs4=C[23],C[28],C[37],C[49]
for i in range(4):
    Obs1,Obs2,Obs3,Obs4=next_situation(Obs1,bool=True),next_situation(Obs2,bool=True),next_situation(Obs3,bool=True),next_situation(Obs4,bool=True)
Observation={t1:Obs1,t2:Obs2,t3:Obs3,t4:Obs4}


### predict some future situations

future=[Model]
obs=[Model]
o=Model
t=0
while t<40:
    disp=False
    if t==10:
        disp=True
    t+=1
    successor,weight=Cata.Neighbors(Model,disp)
    Model=Model.S_prediction(successor,weight)
    if Observation.get(t)!=None:
        o=Observation.get(t)
        Model=Model.S_correction(o)
    obs.append(o)
    future.append(Model)

### show animation

fig=plt.figure(1)
for i in range(len(future)):
    plt.clf()
    ax = fig.add_subplot(121, aspect='equal',title='Prediction t='+str(i))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    for e in future[i].S_map():
        ax.add_artist(e)
    ax_ = fig.add_subplot(122, aspect='equal',title='Last observation')
    ax_.set_xlim(0, 10)
    ax_.set_ylim(0, 10)
    for e in obs[i].S_map():
        ax_.add_artist(e)
    plt.pause(0.4)

plt.show()
