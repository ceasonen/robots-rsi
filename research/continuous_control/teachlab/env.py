"""2-D inertial point-mass control abstraction (not rigid-body robotics)."""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class Task:
    name:str; inertia:float; damping:float; impulse:float; max_accel:float
    horizon:int=100; dt:float=.04
TASKS={
 "impulsive":Task("impulsive",1.0,.12,.85,2.4),
 "sluggish":Task("sluggish",1.8,.22,.45,1.5),
 "precision":Task("precision",1.1,.16,.65,1.8)}
class PointMassEnv:
 def __init__(self,task): self.task=task; self.reset(0)
 def reset(self,seed):
  rng=np.random.default_rng(int(seed)); self.p=rng.uniform(-.12,.12,2); self.v=np.zeros(2)
  self.phase=0; self.t=0; self.shock=rng.uniform(-self.task.impulse,self.task.impulse,2)
  self.done=False; self.success=False; return self.obs()
 @property
 def goal(self): return np.array([.55,-.25]) if self.phase==0 else np.array([-.55,.35])
 def obs(self): return np.array([*self.p,*self.v,float(self.phase),self.t/self.task.horizon,1.],dtype=np.float64)
 def step(self,action):
  if self.done: raise RuntimeError("episode already terminated")
  a=np.asarray(action,dtype=float).reshape(2)
  if not np.isfinite(a).all(): raise ValueError("non-finite action")
  a=np.clip(a,-self.task.max_accel,self.task.max_accel)
  acc=a/self.task.inertia-self.task.damping*self.v
  self.v=np.clip(self.v+self.task.dt*acc,-2.5,2.5); self.p=np.clip(self.p+self.task.dt*self.v,-1.5,1.5)
  self.t+=1
  if self.phase==0 and np.linalg.norm(self.p-np.array([.55,-.25]))<.085:
   self.phase=1; self.v=self.v+self.shock
  err=np.linalg.norm(self.p-self.goal)
  self.success=bool(self.phase==1 and err<.075 and np.linalg.norm(self.v)<.25)
  self.done=self.success or self.t>=self.task.horizon
  return self.obs(),float(self.success),self.done,{"success":self.success}
