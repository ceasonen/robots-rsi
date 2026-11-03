import numpy as np
def teacher(style,env):
 if style=="aggressive": kp,kd=8.0,4.5
 elif style=="damped": kp,kd=5.5,5.0
 else: raise ValueError(style)
 def act(obs):
  return np.clip(kp*(env.goal-obs[:2])-kd*obs[2:4],-env.task.max_accel,env.task.max_accel)
 return act
def parent(env):
 # Read the phase from the evaluated policy observation. A separate
 # parent instance is never advanced by the evaluation loop.
 def act(obs):
  goal=np.array([.55,-.25]) if int(obs[4])==0 else np.array([-.55,.35])
  return np.clip(4.0*(goal-obs[:2])-3.0*obs[2:4],-env.task.max_accel,env.task.max_accel)
 return act
