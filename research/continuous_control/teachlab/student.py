import numpy as np
def features(obs):
 x=np.asarray(obs,dtype=float); p=x[:2];v=x[2:4];phase=x[4];tau=x[5]
 return np.array([1.,*p,*v,phase,tau,phase*p[0],phase*p[1],phase*v[0],phase*v[1],p[0]*p[1]])
class Student:
 def __init__(self,max_accel,ridge=.03): self.max_accel=max_accel;self.ridge=ridge;self.w=None
 def fit(self,rows):
  X=np.stack([features(o) for o,a in rows]);Y=np.stack([a for o,a in rows])
  reg=self.ridge*np.eye(X.shape[1]);reg[0,0]=0;self.w=np.linalg.solve(X.T@X+reg,X.T@Y);return self
 def __call__(self,obs): return np.clip(features(obs)@self.w,-self.max_accel,self.max_accel)
